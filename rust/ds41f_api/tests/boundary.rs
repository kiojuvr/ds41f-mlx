use ds41f_api::{ClientConfig, Ds41fClient, Ds41fError, RuntimeProcess, ShutdownKind};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::process::Command;
use std::thread;
use std::time::Duration;

fn one_shot(response: &'static str, capture: std::sync::mpsc::Sender<String>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = [0u8; 8192];
        let n = stream.read(&mut buf).unwrap();
        capture
            .send(String::from_utf8_lossy(&buf[..n]).into_owned())
            .unwrap();
        stream.write_all(response.as_bytes()).unwrap();
    });
    format!("http://127.0.0.1:{}", addr.port())
}

#[test]
fn health_parses_readiness_without_owning_state() {
    let (tx, rx) = std::sync::mpsc::channel();
    let base = one_shot("HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n{\"status\":\"ready\",\"process_alive\":true,\"model_ready\":true,\"fatal_error\":null}", tx);
    let client = Ds41fClient::new(ClientConfig {
        base_url: base,
        timeout: Duration::from_secs(5),
    })
    .unwrap();
    let h = client.health().unwrap();
    assert_eq!(h.status, "ready");
    assert!(h.process_alive && h.model_ready);
    let req = rx.recv().unwrap();
    assert!(req.starts_with("GET /health HTTP/1.1"));
}

#[test]
fn stateful_session_paths_use_server_session_authority() {
    let (tx, rx) = std::sync::mpsc::channel();
    let base = one_shot(
        "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n{\"id\":\"s1\",\"state\":\"empty\"}",
        tx,
    );
    let client = Ds41fClient::new(ClientConfig {
        base_url: base,
        timeout: Duration::from_secs(5),
    })
    .unwrap();
    let s = client.create_session(Some("s1")).unwrap();
    assert_eq!(s.id, "s1");
    let req = rx.recv().unwrap();
    assert!(req.starts_with("POST /v1/sessions HTTP/1.1"));
    assert!(req.contains("{\"id\":\"s1\"}"));
}

#[test]
fn http_errors_propagate_without_fallback() {
    let (tx, _rx) = std::sync::mpsc::channel();
    let base = one_shot("HTTP/1.1 409 Conflict\r\nConnection: close\r\n\r\n{\"error\":{\"message\":\"active request\"}}", tx);
    let client = Ds41fClient::new(ClientConfig {
        base_url: base,
        timeout: Duration::from_secs(5),
    })
    .unwrap();
    match client.models_raw().unwrap_err() {
        Ds41fError::HttpStatus { status, body } => {
            assert_eq!(status, 409);
            assert!(body.contains("active request"));
        }
        e => panic!("unexpected error {e:?}"),
    }
}

#[test]
fn sse_stream_yields_events_and_drop_cancels_connection() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = [0u8; 4096];
        let _ = stream.read(&mut buf).unwrap();
        stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n")
            .unwrap();
        stream.write_all(b"data: {\"delta\":\"a\"}\n\n").unwrap();
        std::thread::sleep(Duration::from_millis(50));
        let second = stream.write_all(b"data: {\"delta\":\"b\"}\n\n");
        tx.send(second.is_err()).unwrap();
    });
    let client = Ds41fClient::new(ClientConfig {
        base_url: format!("http://127.0.0.1:{}", addr.port()),
        timeout: Duration::from_secs(5),
    })
    .unwrap();
    let mut s = client.chat_completions_stream("{\"stream\":true}").unwrap();
    let first = s.next().unwrap().unwrap();
    assert!(first.data.contains("delta"));
    drop(s);
    let _ = rx.recv_timeout(Duration::from_secs(2));
}

#[test]
fn chunked_body_is_decoded_for_json_endpoints() {
    let (tx, _rx) = std::sync::mpsc::channel();
    let base = one_shot("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n7\r\n{\"ok\":1\r\n1\r\n}\r\n0\r\n\r\n", tx);
    let client = Ds41fClient::new(ClientConfig {
        base_url: base,
        timeout: Duration::from_secs(5),
    })
    .unwrap();
    assert_eq!(client.models_raw().unwrap(), "{\"ok\":1}");
}

#[test]
fn runtime_process_distinguishes_alive_from_model_ready_and_graceful_shutdown() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    let script = r#"
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class H(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        body = b'{"status":"alive","process_alive":true,"model_ready":false,"fatal_error":null}'
        self.send_response(200); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
server = ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1])), H)
server.serve_forever()
"#;
    let mut cmd = Command::new("python3");
    cmd.args(["-c", script, &port.to_string()]);
    let client = Ds41fClient::local(port);
    let mut proc = RuntimeProcess::spawn(cmd, client, Duration::from_secs(10)).unwrap();
    let alive = proc.wait_process_alive(Duration::from_secs(2)).unwrap();
    assert_eq!(alive.status, "alive");
    assert!(!alive.model_ready);
    let ready = proc
        .wait_model_ready(Duration::from_millis(600))
        .unwrap_err()
        .to_string();
    assert!(ready.contains("model ready timeout"), "{ready}");
    let report = proc
        .shutdown(Duration::from_secs(5), Duration::from_secs(2))
        .unwrap();
    assert_eq!(report.kind, ShutdownKind::Graceful);
}

#[test]
fn runtime_process_reports_forced_shutdown_fallback() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    let script = r#"
import signal, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
signal.signal(signal.SIGTERM, signal.SIG_IGN)
class H(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        body = b'{"status":"alive","process_alive":true,"model_ready":false,"fatal_error":null}'
        self.send_response(200); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()
"#;
    let mut cmd = Command::new("python3");
    cmd.args(["-c", script, &port.to_string()]);
    let client = Ds41fClient::local(port);
    let proc = RuntimeProcess::spawn(cmd, client, Duration::from_secs(10)).unwrap();
    let report = proc
        .shutdown(Duration::from_millis(200), Duration::from_secs(2))
        .unwrap();
    assert_eq!(report.kind, ShutdownKind::Forced);
}

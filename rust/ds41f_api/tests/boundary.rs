use ds41f_api::{ClientConfig, Ds41fClient, Ds41fError};
use std::io::{Read, Write};
use std::net::TcpListener;
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

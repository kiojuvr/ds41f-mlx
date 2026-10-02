//! Rust-facing boundary for the qualified ds41f-mlx runtime.
//!
//! This crate is deliberately a loopback HTTP/process-control client.  It does
//! not embed Python, MLX, oMLX, DeepSeek recipe code, KV caches, or prompt
//! conversion logic.  Those remain owned by the M20-qualified ds41f server.

use std::fmt;
use std::io::{BufRead, BufReader, Write};
use std::net::{Shutdown, TcpStream};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Debug)]
pub enum Ds41fError {
    InvalidUrl(String),
    Io(std::io::Error),
    HttpStatus { status: u16, body: String },
    Protocol(String),
    Timeout(String),
    Process(String),
}

impl fmt::Display for Ds41fError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidUrl(s) => write!(f, "invalid ds41f url: {s}"),
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::HttpStatus { status, body } => write!(f, "http status {status}: {body}"),
            Self::Protocol(s) => write!(f, "protocol error: {s}"),
            Self::Timeout(s) => write!(f, "timeout: {s}"),
            Self::Process(s) => write!(f, "process error: {s}"),
        }
    }
}

impl std::error::Error for Ds41fError {}
impl From<std::io::Error> for Ds41fError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, Ds41fError>;

#[derive(Clone, Debug)]
pub struct ClientConfig {
    pub base_url: String,
    pub timeout: Duration,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            base_url: "http://127.0.0.1:8000".into(),
            timeout: Duration::from_secs(1800),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Ds41fClient {
    host: String,
    port: u16,
    timeout: Duration,
}

#[derive(Clone, Debug)]
pub struct Health {
    pub status: String,
    pub process_alive: bool,
    pub model_ready: bool,
    pub fatal_error: Option<String>,
    pub raw_json: String,
}

#[derive(Clone, Debug)]
pub struct Session {
    pub id: String,
    pub raw_json: String,
}

#[derive(Clone, Debug)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: String,
}

impl Ds41fClient {
    pub fn new(config: ClientConfig) -> Result<Self> {
        let (host, port) = parse_loopback_http_url(&config.base_url)?;
        Ok(Self {
            host,
            port,
            timeout: config.timeout,
        })
    }

    pub fn local(port: u16) -> Self {
        Self {
            host: "127.0.0.1".into(),
            port,
            timeout: Duration::from_secs(1800),
        }
    }

    pub fn health(&self) -> Result<Health> {
        let body = self.request_json("GET", "/health", None)?;
        Ok(Health {
            status: json_string_field(&body, "status").unwrap_or_default(),
            process_alive: json_bool_field(&body, "process_alive").unwrap_or(false),
            model_ready: json_bool_field(&body, "model_ready").unwrap_or(false),
            fatal_error: json_nullable_string_field(&body, "fatal_error"),
            raw_json: body,
        })
    }

    pub fn models_raw(&self) -> Result<String> {
        self.request_json("GET", "/v1/models", None)
    }

    pub fn chat_completions_raw(&self, body_json: &str) -> Result<String> {
        self.request_json("POST", "/v1/chat/completions", Some(body_json))
    }

    pub fn chat_completions_stream(&self, body_json: &str) -> Result<SseStream> {
        self.request_stream("POST", "/v1/chat/completions", Some(body_json))
    }

    pub fn create_session(&self, id: Option<&str>) -> Result<Session> {
        let body = match id {
            Some(id) => format!("{{\"id\":\"{}\"}}", escape_json(id)),
            None => "{}".into(),
        };
        let raw = self.request_json("POST", "/v1/sessions", Some(&body))?;
        let sid = json_string_field(&raw, "id").ok_or_else(|| {
            Ds41fError::Protocol("create_session response did not include id".into())
        })?;
        Ok(Session {
            id: sid,
            raw_json: raw,
        })
    }

    pub fn get_session_raw(&self, id: &str) -> Result<String> {
        self.request_json("GET", &format!("/v1/sessions/{}", path_component(id)), None)
    }

    pub fn close_session_raw(&self, id: &str) -> Result<String> {
        self.request_json(
            "DELETE",
            &format!("/v1/sessions/{}", path_component(id)),
            None,
        )
    }

    pub fn session_chat_completions_raw(&self, id: &str, body_json: &str) -> Result<String> {
        self.request_json(
            "POST",
            &format!("/v1/sessions/{}/chat/completions", path_component(id)),
            Some(body_json),
        )
    }

    pub fn session_chat_completions_stream(&self, id: &str, body_json: &str) -> Result<SseStream> {
        self.request_stream(
            "POST",
            &format!("/v1/sessions/{}/chat/completions", path_component(id)),
            Some(body_json),
        )
    }

    pub fn persist_session_raw(&self, id: &str, artifact_root: Option<&str>) -> Result<String> {
        let body = match artifact_root {
            Some(p) => format!("{{\"artifact_root\":\"{}\"}}", escape_json(p)),
            None => "{}".into(),
        };
        self.request_json(
            "POST",
            &format!("/v1/sessions/{}/persist", path_component(id)),
            Some(&body),
        )
    }

    pub fn restore_session_raw(&self, artifact_path: &str, id: Option<&str>) -> Result<String> {
        let mut body = format!("{{\"artifact_path\":\"{}\"", escape_json(artifact_path));
        if let Some(id) = id {
            body.push_str(&format!(",\"id\":\"{}\"", escape_json(id)));
        }
        body.push('}');
        self.request_json("POST", "/v1/sessions/restore", Some(&body))
    }

    pub fn raw_json(&self, method: &str, path: &str, body_json: Option<&str>) -> Result<String> {
        self.request_json(method, path, body_json)
    }

    fn connect(&self) -> Result<TcpStream> {
        let stream = TcpStream::connect((self.host.as_str(), self.port))?;
        stream.set_read_timeout(Some(self.timeout))?;
        stream.set_write_timeout(Some(self.timeout))?;
        Ok(stream)
    }

    fn write_request(
        &self,
        stream: &mut TcpStream,
        method: &str,
        path: &str,
        body: Option<&str>,
    ) -> Result<()> {
        let body = body.unwrap_or("");
        let content_headers = if body.is_empty() {
            String::new()
        } else {
            format!(
                "Content-Type: application/json\r\nContent-Length: {}\r\n",
                body.as_bytes().len()
            )
        };
        let req = format!("{method} {path} HTTP/1.1\r\nHost: {}:{}\r\nAccept: application/json, text/event-stream\r\nConnection: close\r\n{content_headers}\r\n{body}", self.host, self.port);
        stream.write_all(req.as_bytes())?;
        stream.flush()?;
        Ok(())
    }

    fn request_json(&self, method: &str, path: &str, body: Option<&str>) -> Result<String> {
        let mut stream = self.connect()?;
        self.write_request(&mut stream, method, path, body)?;
        let response = read_http_response(stream)?;
        if !(200..300).contains(&response.status) {
            return Err(Ds41fError::HttpStatus {
                status: response.status,
                body: response.body,
            });
        }
        Ok(response.body)
    }

    fn request_stream(&self, method: &str, path: &str, body: Option<&str>) -> Result<SseStream> {
        let mut stream = self.connect()?;
        self.write_request(&mut stream, method, path, body)?;
        let mut reader = BufReader::new(stream);
        let (status, headers) = read_status_and_headers(&mut reader)?;
        if !(200..300).contains(&status) {
            let body = read_body_from_reader(&mut reader, &headers)?;
            return Err(Ds41fError::HttpStatus { status, body });
        }
        let chunked = headers
            .iter()
            .any(|(k, v)| k == "transfer-encoding" && v.to_ascii_lowercase().contains("chunked"));
        let body = if chunked {
            StreamBody::Chunked(ChunkedLineReader::new(reader))
        } else {
            StreamBody::Plain(reader)
        };
        Ok(SseStream {
            body,
            buffer: String::new(),
            done: false,
        })
    }
}

pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

pub struct SseStream {
    body: StreamBody,
    buffer: String,
    done: bool,
}

enum StreamBody {
    Plain(BufReader<TcpStream>),
    Chunked(ChunkedLineReader<BufReader<TcpStream>>),
}

impl StreamBody {
    fn read_line(&mut self, s: &mut String) -> std::io::Result<usize> {
        match self {
            Self::Plain(r) => r.read_line(s),
            Self::Chunked(r) => r.read_line(s),
        }
    }
    fn shutdown(&mut self) {
        match self {
            Self::Plain(r) => {
                let _ = r.get_ref().shutdown(Shutdown::Both);
            }
            Self::Chunked(r) => {
                let _ = r.inner.get_ref().shutdown(Shutdown::Both);
            }
        }
    }
}

struct ChunkedLineReader<R: BufRead> {
    inner: R,
    remaining: usize,
    done: bool,
}
impl<R: BufRead> ChunkedLineReader<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            remaining: 0,
            done: false,
        }
    }
    fn next_byte(&mut self) -> std::io::Result<Option<u8>> {
        if self.done {
            return Ok(None);
        }
        if self.remaining == 0 {
            let mut size_line = String::new();
            if self.inner.read_line(&mut size_line)? == 0 {
                self.done = true;
                return Ok(None);
            }
            let size_hex = size_line.trim().split(';').next().unwrap_or("");
            self.remaining = usize::from_str_radix(size_hex, 16).map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "bad chunk size")
            })?;
            if self.remaining == 0 {
                self.done = true;
                return Ok(None);
            }
        }
        let mut b = [0u8; 1];
        self.inner.read_exact(&mut b)?;
        self.remaining -= 1;
        if self.remaining == 0 {
            let mut crlf = [0u8; 2];
            self.inner.read_exact(&mut crlf)?;
        }
        Ok(Some(b[0]))
    }
    fn read_line(&mut self, s: &mut String) -> std::io::Result<usize> {
        let start = s.len();
        while let Some(b) = self.next_byte()? {
            s.push(b as char);
            if b == b'\n' {
                break;
            }
        }
        Ok(s.len() - start)
    }
}

impl SseStream {
    pub fn close(&mut self) {
        self.body.shutdown();
        self.done = true;
    }
}

impl Drop for SseStream {
    fn drop(&mut self) {
        self.close();
    }
}

impl Iterator for SseStream {
    type Item = Result<SseEvent>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let mut event: Option<String> = None;
        let mut data_lines: Vec<String> = Vec::new();
        loop {
            self.buffer.clear();
            match self.body.read_line(&mut self.buffer) {
                Ok(0) => {
                    self.done = true;
                    return if data_lines.is_empty() {
                        None
                    } else {
                        Some(Ok(SseEvent {
                            event,
                            data: data_lines.join("\n"),
                        }))
                    };
                }
                Ok(_) => {
                    let line = self.buffer.trim_end_matches(['\r', '\n']);
                    if line.is_empty() {
                        if data_lines.is_empty() {
                            continue;
                        }
                        let data = data_lines.join("\n");
                        if data == "[DONE]" {
                            self.done = true;
                        }
                        return Some(Ok(SseEvent { event, data }));
                    }
                    if let Some(rest) = line.strip_prefix("event:") {
                        event = Some(rest.trim_start().to_string());
                    } else if let Some(rest) = line.strip_prefix("data:") {
                        data_lines.push(rest.trim_start().to_string());
                    }
                }
                Err(e) => {
                    self.done = true;
                    return Some(Err(Ds41fError::Io(e)));
                }
            }
        }
    }
}

pub struct RuntimeProcess {
    child: Child,
    client: Ds41fClient,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShutdownKind {
    Graceful,
    Forced,
    AlreadyExited,
}

#[derive(Clone, Debug)]
pub struct ShutdownReport {
    pub kind: ShutdownKind,
    pub exit_status: Option<String>,
}

impl RuntimeProcess {
    pub fn spawn(
        mut command: Command,
        client: Ds41fClient,
        process_alive_timeout: Duration,
    ) -> Result<Self> {
        command.stdout(Stdio::null()).stderr(Stdio::null());
        let child = command.spawn().map_err(Ds41fError::Io)?;
        let mut proc = Self { child, client };
        proc.wait_process_alive(process_alive_timeout)?;
        Ok(proc)
    }

    pub fn spawn_python_module(
        python: &str,
        host: &str,
        port: u16,
        process_alive_timeout: Duration,
    ) -> Result<Self> {
        let mut cmd = Command::new(python);
        cmd.args([
            "-m",
            "ds41f_mlx.serve",
            "--host",
            host,
            "--port",
            &port.to_string(),
        ]);
        Self::spawn(cmd, Ds41fClient::local(port), process_alive_timeout)
    }

    pub fn spawn_python_module_from_env(
        host: &str,
        port: u16,
        process_alive_timeout: Duration,
    ) -> Result<Self> {
        let python = std::env::var("DS41F_PYTHON")
            .or_else(|_| std::env::var("DS41F_RUNTIME_PYTHON"))
            .unwrap_or_else(|_| "python3".to_string());
        Self::spawn_python_module(&python, host, port, process_alive_timeout)
    }

    pub fn client(&self) -> &Ds41fClient {
        &self.client
    }

    pub fn child_id(&self) -> u32 {
        self.child.id()
    }

    pub fn wait_process_alive(&mut self, timeout: Duration) -> Result<Health> {
        self.wait_for_health(
            timeout,
            |h| h.status == "alive" || h.status == "ready",
            "process alive",
        )
    }

    pub fn wait_model_ready(&mut self, timeout: Duration) -> Result<Health> {
        self.wait_for_health(
            timeout,
            |h| h.status == "ready" && h.model_ready,
            "model ready",
        )
    }

    pub fn wait_ready(&mut self, timeout: Duration) -> Result<Health> {
        self.wait_model_ready(timeout)
    }

    fn wait_for_health<F>(
        &mut self,
        timeout: Duration,
        mut accept: F,
        label: &str,
    ) -> Result<Health>
    where
        F: FnMut(&Health) -> bool,
    {
        let deadline = Instant::now() + timeout;
        let mut last = String::new();
        while Instant::now() < deadline {
            if let Some(status) = self.child.try_wait()? {
                return Err(Ds41fError::Process(format!(
                    "server exited before {label}: {status}"
                )));
            }
            match self.client.health() {
                Ok(h) if h.status == "unavailable" || h.fatal_error.is_some() => {
                    return Err(Ds41fError::Process(format!(
                        "server fatal health: {}",
                        h.raw_json
                    )));
                }
                Ok(h) if accept(&h) => return Ok(h),
                Ok(h) => last = h.raw_json,
                Err(e) => last = e.to_string(),
            }
            std::thread::sleep(Duration::from_millis(250));
        }
        Err(Ds41fError::Timeout(format!(
            "server {label} timeout; last={last}"
        )))
    }

    pub fn terminate_gracefully(&mut self, timeout: Duration) -> Result<Option<String>> {
        if let Some(status) = self.child.try_wait()? {
            return Ok(Some(status.to_string()));
        }
        send_sigterm(self.child.id())?;
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if let Some(status) = self.child.try_wait()? {
                return Ok(Some(status.to_string()));
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        Ok(None)
    }

    pub fn shutdown(
        mut self,
        graceful_timeout: Duration,
        forced_timeout: Duration,
    ) -> Result<ShutdownReport> {
        if let Some(status) = self.child.try_wait()? {
            return Ok(ShutdownReport {
                kind: ShutdownKind::AlreadyExited,
                exit_status: Some(status.to_string()),
            });
        }
        if let Some(status) = self.terminate_gracefully(graceful_timeout)? {
            return Ok(ShutdownReport {
                kind: ShutdownKind::Graceful,
                exit_status: Some(status),
            });
        }
        self.child.kill().ok();
        let deadline = Instant::now() + forced_timeout;
        while Instant::now() < deadline {
            if let Some(status) = self.child.try_wait()? {
                return Ok(ShutdownReport {
                    kind: ShutdownKind::Forced,
                    exit_status: Some(status.to_string()),
                });
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        Err(Ds41fError::Timeout("server forced shutdown timeout".into()))
    }
}

impl Drop for RuntimeProcess {
    fn drop(&mut self) {
        let _ = self.terminate_gracefully(Duration::from_secs(2));
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn send_sigterm(pid: u32) -> Result<()> {
    let status = Command::new("/bin/kill")
        .args(["-TERM", &pid.to_string()])
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(Ds41fError::Process(format!(
            "failed to send SIGTERM to pid {pid}: {status}"
        )))
    }
}

fn read_http_response(stream: TcpStream) -> Result<HttpResponse> {
    let mut reader = BufReader::new(stream);
    let (status, headers) = read_status_and_headers(&mut reader)?;
    let body = read_body_from_reader(&mut reader, &headers)?;
    Ok(HttpResponse {
        status,
        headers,
        body,
    })
}

fn read_status_and_headers<R: BufRead>(reader: &mut R) -> Result<(u16, Vec<(String, String)>)> {
    let mut status_line = String::new();
    reader.read_line(&mut status_line)?;
    let status = status_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| Ds41fError::Protocol(format!("bad status line: {status_line:?}")))?
        .parse::<u16>()
        .map_err(|_| Ds41fError::Protocol(format!("bad status line: {status_line:?}")))?;
    let mut headers = Vec::new();
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        if let Some((k, v)) = trimmed.split_once(':') {
            headers.push((k.trim().to_ascii_lowercase(), v.trim().to_string()));
        }
    }
    Ok((status, headers))
}

fn read_body_from_reader<R: BufRead>(
    reader: &mut R,
    headers: &[(String, String)],
) -> Result<String> {
    let chunked = headers
        .iter()
        .any(|(k, v)| k == "transfer-encoding" && v.to_ascii_lowercase().contains("chunked"));
    if chunked {
        return read_chunked_body(reader);
    }
    if let Some(len) = headers
        .iter()
        .find(|(k, _)| k == "content-length")
        .and_then(|(_, v)| v.parse::<usize>().ok())
    {
        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf)?;
        return Ok(String::from_utf8_lossy(&buf).into_owned());
    }
    let mut s = String::new();
    reader.read_to_string(&mut s)?;
    Ok(s)
}

fn read_chunked_body<R: BufRead>(reader: &mut R) -> Result<String> {
    let mut out = Vec::new();
    loop {
        let mut size_line = String::new();
        reader.read_line(&mut size_line)?;
        let size_hex = size_line.trim().split(';').next().unwrap_or("");
        let size = usize::from_str_radix(size_hex, 16)
            .map_err(|_| Ds41fError::Protocol(format!("bad chunk size: {size_line:?}")))?;
        if size == 0 {
            break;
        }
        let mut buf = vec![0u8; size];
        reader.read_exact(&mut buf)?;
        out.extend_from_slice(&buf);
        let mut crlf = [0u8; 2];
        reader.read_exact(&mut crlf)?;
    }
    Ok(String::from_utf8_lossy(&out).into_owned())
}

fn parse_loopback_http_url(url: &str) -> Result<(String, u16)> {
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| Ds41fError::InvalidUrl(url.into()))?;
    let authority = rest.split('/').next().unwrap_or(rest);
    let (host, port_s) = authority
        .rsplit_once(':')
        .ok_or_else(|| Ds41fError::InvalidUrl(url.into()))?;
    let port = port_s
        .parse::<u16>()
        .map_err(|_| Ds41fError::InvalidUrl(url.into()))?;
    if host != "127.0.0.1" && host != "localhost" {
        return Err(Ds41fError::InvalidUrl(
            "M21 supports loopback HTTP only".into(),
        ));
    }
    Ok((host.into(), port))
}

fn json_string_field(body: &str, key: &str) -> Option<String> {
    let needle = format!("\"{}\"", key);
    let pos = body.find(&needle)?;
    let after = &body[pos + needle.len()..];
    let colon = after.find(':')?;
    parse_json_string(after[colon + 1..].trim_start())
}

fn json_nullable_string_field(body: &str, key: &str) -> Option<String> {
    json_string_field(body, key)
}

fn json_bool_field(body: &str, key: &str) -> Option<bool> {
    let needle = format!("\"{}\"", key);
    let pos = body.find(&needle)?;
    let after = &body[pos + needle.len()..];
    let colon = after.find(':')?;
    let v = after[colon + 1..].trim_start();
    if v.starts_with("true") {
        Some(true)
    } else if v.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

fn parse_json_string(s: &str) -> Option<String> {
    let mut chars = s.chars();
    if chars.next()? != '"' {
        return None;
    }
    let mut out = String::new();
    let mut esc = false;
    for c in chars {
        if esc {
            out.push(match c {
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                '"' => '"',
                '\\' => '\\',
                other => other,
            });
            esc = false;
        } else if c == '\\' {
            esc = true;
        } else if c == '"' {
            return Some(out);
        } else {
            out.push(c);
        }
    }
    None
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}
fn path_component(s: &str) -> String {
    let mut out = String::new();
    for b in s.as_bytes() {
        match *b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(*b as char)
            }
            other => out.push_str(&format!("%{other:02X}")),
        }
    }
    out
}

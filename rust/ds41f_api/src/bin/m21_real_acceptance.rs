use ds41f_api::{ClientConfig, Ds41fClient, RuntimeProcess, ShutdownKind};
use std::net::TcpListener;
use std::time::{Duration, Instant};

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind free port");
    listener.local_addr().unwrap().port()
}

fn chat_body(messages: &str, max_tokens: u32, stream: bool) -> String {
    format!(
        "{{\"model\":\"deepseek-v4.1-flash\",\"messages\":{messages},\"temperature\":0,\"reasoning_effort\":\"none\",\"max_tokens\":{max_tokens},\"stream\":{stream}}}"
    )
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn content_prefix(raw: &str) -> String {
    if let Some(content) = json_string_field(raw, "content") {
        return content.chars().take(160).collect();
    }
    raw.chars().take(160).collect()
}

fn json_string_field(raw: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\":");
    let pos = raw.find(&needle)?;
    parse_json_string(raw[pos + needle.len()..].trim_start())
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let port = std::env::var("DS41F_M21_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(free_port);
    let python = std::env::var("DS41F_PYTHON")
        .or_else(|_| std::env::var("DS41F_RUNTIME_PYTHON"))
        .unwrap_or_else(|_| "python3".into());
    let start = Instant::now();
    let mut steps: Vec<String> = Vec::new();
    println!("M21 real acceptance starting: python={python} port={port}");
    let mut proc =
        RuntimeProcess::spawn_python_module(&python, "127.0.0.1", port, Duration::from_secs(180))?;
    let h0 = proc.client().health()?;
    steps.push(format!(
        "startup_health status={} model_ready={}",
        h0.status, h0.model_ready
    ));
    if h0.status != "alive" && h0.status != "ready" {
        return Err(format!("unexpected startup health {}", h0.raw_json).into());
    }

    let models = proc.client().models_raw()?;
    steps.push(format!("models {}", content_prefix(&models)));

    let stateless_body = chat_body(
        "[{\"role\":\"user\",\"content\":\"Answer with exactly one short sentence: 2+2?\"}]",
        16,
        false,
    );
    let stateless = proc.client().chat_completions_raw(&stateless_body)?;
    steps.push(format!("stateless {}", content_prefix(&stateless)));
    let h1 = proc.wait_model_ready(Duration::from_secs(30))?;
    steps.push(format!(
        "post_inference_health status={} model_ready={}",
        h1.status, h1.model_ready
    ));

    let stream_body = chat_body(
        "[{\"role\":\"user\",\"content\":\"Count one two three, very briefly.\"}]",
        32,
        true,
    );
    let mut stream = proc.client().chat_completions_stream(&stream_body)?;
    if let Some(ev) = stream.next() {
        steps.push(format!(
            "stream_first_event {}",
            ev?.data.chars().take(160).collect::<String>()
        ));
    }
    drop(stream);
    std::thread::sleep(Duration::from_secs(1));
    let recovery = proc.client().models_raw()?;
    steps.push(format!(
        "post_cancel_recovery_models {}",
        content_prefix(&recovery)
    ));

    let session = proc.client().create_session(None)?;
    steps.push(format!("session_created id={}", session.id));
    let first_messages = "[{\"role\":\"user\",\"content\":\"Say a short greeting.\"}]";
    let first_body = chat_body(first_messages, 16, false);
    let first = proc
        .client()
        .session_chat_completions_raw(&session.id, &first_body)?;
    steps.push(format!("stateful_first {}", content_prefix(&first)));
    let assistant = json_escape(
        &json_string_field(&first, "content").unwrap_or_else(|| content_prefix(&first)),
    );
    let second_messages = format!("[{{\"role\":\"user\",\"content\":\"Say a short greeting.\"}},{{\"role\":\"assistant\",\"content\":\"{assistant}\"}},{{\"role\":\"user\",\"content\":\"Now say goodbye briefly.\"}}]");
    let second_body = chat_body(&second_messages, 16, false);
    let second = proc
        .client()
        .session_chat_completions_raw(&session.id, &second_body)?;
    steps.push(format!("stateful_second {}", content_prefix(&second)));
    let closed = proc.client().close_session_raw(&session.id)?;
    steps.push(format!("session_closed {}", content_prefix(&closed)));

    let missing = Ds41fClient::new(ClientConfig {
        base_url: format!("http://127.0.0.1:{port}"),
        timeout: Duration::from_secs(30),
    })?
    .get_session_raw("definitely-missing-session");
    match missing {
        Err(ds41f_api::Ds41fError::HttpStatus { status: 404, .. }) => {
            steps.push("missing_session_404 propagated".into())
        }
        other => return Err(format!("expected 404 propagation, got {other:?}").into()),
    }

    let report = proc.shutdown(Duration::from_secs(120), Duration::from_secs(10))?;
    if report.kind != ShutdownKind::Graceful && report.kind != ShutdownKind::AlreadyExited {
        return Err(format!("server did not shut down gracefully: {:?}", report).into());
    }
    steps.push(format!(
        "shutdown {:?} {:?}",
        report.kind, report.exit_status
    ));

    println!(
        "M21_REAL_ACCEPTANCE_PASS duration_s={:.3}",
        start.elapsed().as_secs_f64()
    );
    for s in &steps {
        println!("STEP {s}");
    }
    Ok(())
}

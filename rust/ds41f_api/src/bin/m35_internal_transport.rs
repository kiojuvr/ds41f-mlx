//! Qualification client only: uses the existing HTTP/SSE client unchanged.
use ds41f_api::{Ds41fClient, Result};
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let port: u16 = args[1].parse().expect("port");
    let body = std::fs::read_to_string(&args[3])?;
    let client = Ds41fClient::local(port);
    let mut stream = client.session_chat_completions_stream(&args[2], &body)?;
    for _ in 0..40 {
        if let Some(event) = stream.next() {
            let event = event?;
            println!("{}", event.data);
        } else { break; }
    }
    drop(stream); // Existing Shutdown::Both, no runtime/cache authority here.
    println!("{}", client.health()?.raw_json);
    Ok(())
}

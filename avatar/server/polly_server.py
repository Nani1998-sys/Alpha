#!/usr/bin/env python3
"""Text-to-speech endpoint for the avatar: Amazon Polly audio + viseme timing in one call.

    POST /api/tts   {"text": "Hello", "voice": "Kajal"}          (voice is optional)
    ->  200 {"audio": "<base64 mp3>", "marks": [{"time": 6, "type": "viseme", "value": "k"}, ...],
             "voice": "Kajal", "engine": "neural"}

Polly is asked twice for the same text: once for the MP3, once for the speech marks
(visemes and words). Both runs use the same voice and engine, so their timings line up.
The browser plays the MP3 and moves the mouth from the marks (see web/polly-lipsync.js).

Run it next to the demo page (serves the avatar folder too):
    pip install boto3
    AWS_REGION=us-east-1 python avatar/server/polly_server.py --port 8080
    open http://localhost:8080/web/

AWS credentials come from the usual places (environment, ~/.aws, instance role). The IAM
policy only needs "polly:SynthesizeSpeech". In your own backend, call synthesize() from your
existing request handler instead of running this server.
"""

import argparse
import base64
import functools
import http.server
import json
import os
import sys

DEFAULT_VOICE = os.environ.get("POLLY_VOICE", "Kajal")        # Indian English, neural
DEFAULT_ENGINE = os.environ.get("POLLY_ENGINE", "neural")
MAX_CHARS = 3000                                               # Polly's limit for plain text


def synthesize(text, voice=DEFAULT_VOICE, engine=DEFAULT_ENGINE, client=None):
    """Return (mp3_bytes, marks) for `text`. `marks` is a list of Polly speech-mark dicts."""
    if not text or not text.strip():
        raise ValueError("text is empty")
    if len(text) > MAX_CHARS:
        raise ValueError("text is longer than %d characters" % MAX_CHARS)
    if client is None:
        import boto3
        client = boto3.client("polly")
    common = {"Text": text, "VoiceId": voice, "Engine": engine, "TextType": "text"}
    audio = client.synthesize_speech(OutputFormat="mp3", **common)["AudioStream"].read()
    raw = client.synthesize_speech(OutputFormat="json", SpeechMarkTypes=["viseme", "word"], **common)["AudioStream"].read()
    marks = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    return audio, marks


class Handler(http.server.SimpleHTTPRequestHandler):
    polly_client = None

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        if self.path.split("?")[0] != "/api/tts":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > 64 * 1024:
                raise ValueError("request too large")
            body = json.loads(self.rfile.read(length) or b"{}")
            voice = body.get("voice") or DEFAULT_VOICE
            engine = body.get("engine") or DEFAULT_ENGINE
            audio, marks = synthesize(body.get("text", ""), voice, engine, client=self.polly_client)
            self._json(200, {"audio": base64.b64encode(audio).decode("ascii"), "marks": marks,
                             "voice": voice, "engine": engine})
        except ValueError as error:
            self._json(400, {"error": str(error)})
        except Exception as error:  # boto errors, missing credentials, ...
            self.log_error("TTS failed: %r", error)
            self._json(502, {"error": "text-to-speech failed: %s" % type(error).__name__})

    def _json(self, status, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    parser = argparse.ArgumentParser(description="Serve the avatar demo with an Amazon Polly /api/tts endpoint")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
                        help="folder to serve (default: the avatar folder)")
    args = parser.parse_args()
    handler = functools.partial(Handler, directory=os.path.abspath(args.root))
    server = http.server.ThreadingHTTPServer((args.host, args.port), handler)
    print("Avatar demo on http://%s:%d/web/  (voice %s, engine %s)" % (args.host, args.port, DEFAULT_VOICE, DEFAULT_ENGINE))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

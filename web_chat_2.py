# Copyright 2026 Alexander Komarov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# web_chat_2.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Simple web chat UI used by Iyye."""

from __future__ import annotations   # <-- postpone evaluation of type hints

from flask import Flask, request, jsonify, render_template_string, redirect

import threading
from collections import deque
from typing import Optional, Any

app = Flask(__name__)
_MAX_MESSAGES = 500
messages: deque[str] = deque(maxlen=_MAX_MESSAGES)

# --------------------------------------------------------------------------- #
# Global placeholder that will be set by the brain once it creates its queue.
# It is *optional* – if the UI runs without a brain we simply ignore pushes.
# --------------------------------------------------------------------------- #
_web_chat_queue: Optional["BaseSensorQueue"] = None   # type: ignore[name-defined]

def attach_sensor(queue):
    """
    Called from ``IyyeBrain`` right after it creates its internal
    ``web_chat`` sensor.  The Flask handlers will use this object to push
    incoming chat messages into the brain’s queue.
    """
    global _web_chat_queue
    _web_chat_queue = queue

# --------------------------------------------------------------------------- #
# HTML template for the chat interface (unchanged)
# --------------------------------------------------------------------------- #
HTML_TEMPLATE = '''
<!doctype html>
<html>
<head><title>Web Chat</title></head>
<body>
<h1>Web Chat</h1>
<form id="chatForm">
    <input type="text" id="message" name="message"
           placeholder="Type your message here..." required>
    <button type="submit">Send</button>
</form>

<div id="messages"><h2>Messages:</h2><ul id="msgList">
{% for message in messages %}
  <li>{{ message }}</li>
{% endfor %}
</ul></div>

<script>
let lastCount = {{ messages|length }};

function refreshMessages() {
    fetch('/messages')
      .then(r => r.json())
      .then(data => {
          if (data.length !== lastCount) {
              const ul = document.getElementById('msgList');
              ul.innerHTML = '';
              data.forEach(m => {
                  const li = document.createElement('li');
                  li.textContent = m;
                  ul.appendChild(li);
              });
              lastCount = data.length;
              if (ul.lastElementChild)
                  ul.lastElementChild.scrollIntoView({behavior:'smooth'});
          }
      })
      .catch(() => {});
}

setInterval(refreshMessages, 2000);

document.getElementById('chatForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const msg = document.getElementById('message').value;
    fetch('/chat',{method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({message:msg})})
      .then(r=>r.json())
      .then(d=>{
          if (d.status==='success'){
              document.getElementById('message').value='';
          }
      });
});
</script>
</body>
</html>
'''

# --------------------------------------------------------------------------- #
# Flask routes
# --------------------------------------------------------------------------- #

@app.route("/", methods=["GET"])
def index():
    """Redirect the root URL to the chat UI."""
    return redirect("/chat")

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        message: str | None = data.get('message')
        if not message:
            return jsonify({"status": "error", "message": "Invalid input"}), 400

        # Store for UI rendering
        messages.append(message)

        # Push into brain queue (if the brain is running)
        if _web_chat_queue is not None:
            try:
                _web_chat_queue.push(message)   # <-- creates a DEBUG log entry in Brain
            except Exception as exc:  # pragma: no‑cover – UI must never crash
                app.logger.error("Failed to push chat into brain queue: %s", exc)

        return jsonify({"status": "success", "message": message}), 200

    # GET → render page with accumulated messages
    return render_template_string(HTML_TEMPLATE, messages=messages)


@app.route('/messages', methods=['GET'])
def get_messages_json():
    """Return current messages list as JSON for polling."""
    return jsonify(list(messages))


# --------------------------------------------------------------------------- #
# Helper functions used by ``main_loop.py``
# --------------------------------------------------------------------------- #

def run_web_chat():
    """Run the Flask development server (blocking)."""
    app.run(host="127.0.0.1", port=5000)

def start_web_chat():
    """
    Start the Flask UI in a *daemon* thread so that it does not block
    the main Iyye loop.
    """
    threading.Thread(target=run_web_chat, daemon=True).start()

def get_messages() -> list[str]:
    """Return all chat messages (used by tests / demos)."""
    return list(messages)

def broadcast_debug(text: str) -> None:
    """
    Push debug text into the messages list so it appears in the web UI.
    Called by WebChatActuator to display brain output.
    """
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    messages.append(f"[{ts}] [Iyye] {text}")
    app.logger.info("Broadcast: %s", text)


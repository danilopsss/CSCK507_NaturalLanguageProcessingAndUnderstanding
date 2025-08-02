var wsAttention = new WebSocket("ws://localhost:8000/attention");
var wsNoAttention = new WebSocket("ws://localhost:8000/noAttention");

function displayMessage(message, type) {
    const chatBox = document.getElementById('chat-box');
    const messageElement = document.createElement('div');
    messageElement.classList.add('chat-message', type === 'user' ? 'user-message' : 'server-message');
    messageElement.textContent = message;
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;
}

wsAttention.onopen = function(event) {
    console.log("Attention websocket connected!");
};
wsAttention.onmessage = function(event) {
    displayMessage(event.data, "server-message");
};

wsNoAttention.onopen = function(event) {
    console.log("No Attention websocket connected!");
};
wsNoAttention.onmessage = function(event) {
    displayMessage(event.data, "server-message");
};
function displayMessageInBox(message, type, boxId) {
    const chatBox = document.getElementById(boxId);
    const messageElement = document.createElement('div');
    messageElement.classList.add('chat-message', type === 'user' ? 'user-message' : 'server-message');
    messageElement.textContent = message;
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function displayMessage(message, type) {
    const attentionInput = document.getElementById('attention-input');
    const noAttentionInput = document.getElementById('no-attention-input');
    if (type === 'user') {
        if (document.activeElement === attentionInput) {
            displayMessageInBox(message, type, 'chat-box-2');
        } else if (document.activeElement === noAttentionInput) {
            displayMessageInBox(message, type, 'chat-box-1');
        }
    } else if (type === 'server-message') {
        if (window.lastServer === 'attention') {
            displayMessageInBox(message, type, 'chat-box-2');
        } else if (window.lastServer === 'noAttention') {
            displayMessageInBox(message, type, 'chat-box-1');
        }
    }
}

wsAttention.onmessage = function(event) {
    window.lastServer = 'attention';
    displayMessage(event.data, "server-message");
};
wsNoAttention.onmessage = function(event) {
    window.lastServer = 'noAttention';
    displayMessage(event.data, "server-message");
};
document.getElementById('attention-input').addEventListener('keydown', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault();
        const userMessage = this.value.trim();
        if (userMessage === '') return;
        displayMessage(userMessage, 'user');
        wsAttention.send(userMessage);
        this.value = '';
    }
});

document.getElementById('no-attention-input').addEventListener('keydown', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault();
        const userMessage = this.value.trim();
        if (userMessage === '') return;
        displayMessage(userMessage, 'user');
        wsNoAttention.send(userMessage);
        this.value = '';
    }
});

document.getElementById('attention-send-button').addEventListener('click', function() {
    const userInput = document.getElementById('attention-input');
    const userMessage = userInput.value.trim();
    if (userMessage === '') return;
    displayMessage(userMessage, 'user');
    wsAttention.send(userMessage);
    userInput.value = '';
});

document.getElementById('no-attention-send-button').addEventListener('click', function() {
    const userInput = document.getElementById('no-attention-input');
    const userMessage = userInput.value.trim();
    if (userMessage === '') return;
    displayMessage(userMessage, 'user');
    wsNoAttention.send(userMessage);
    userInput.value = '';
});

var ws = new WebSocket("ws://192.168.56.1:8000/ws");

function displayMessage(message, type) {
    const chatBox = document.getElementById('chat-box');
    const messageElement = document.createElement('div');
    messageElement.classList.add('chat-message', type === 'user' ? 'user-message' : 'server-message');
    messageElement.textContent = message;
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;
}

ws.onopen = function(event) {
    console.log("websocket connected!")
};


ws.onmessage = function(event) {
    displayMessage(event.data, "server-message")
};

function sendMessage(event) {
    var input = document.getElementById("messageText")
    ws.send(input.value)
    input.value = ''
    event.preventDefault()
}

document.querySelector("input[type='text']").addEventListener('keydown', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault();
        const userMessage = this.value.trim();

        if (userMessage === '') return;

        displayMessage(userMessage, 'user');
        ws.send(userMessage);

        this.value = '';
    }
});

document.getElementById('send-button').addEventListener('click', function() {
    const userInput = document.getElementById('user-input');
    const userMessage = userInput.value;

    if (userMessage.trim() === '') return;

    displayMessage(userMessage, 'user');
    ws.send(userMessage)

    userInput.value = '';
});

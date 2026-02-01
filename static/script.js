const chatWindow = document.getElementById('chat-window');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const loading = document.getElementById('loading');
const themeToggle = document.getElementById('theme-toggle');

// Toggle Dark/Light Mode
themeToggle.addEventListener('click', () => {
    document.body.classList.toggle('dark-theme');
    const isDark = document.body.classList.contains('dark-theme');
    themeToggle.innerHTML = isDark ? '<i class="fas fa-sun"></i> Theme' : '<i class="fas fa-moon"></i> Theme';
});

// Append messages to UI
function appendMessage(role, text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}-message`;
    msgDiv.innerText = text;
    chatWindow.appendChild(msgDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

// Main function to call API
async function handleQuery() {
    const query = userInput.value.trim();
    if (!query) return;

    // 1. UI Update: User Message
    appendMessage('user', query);
    userInput.value = '';
    
    // 2. Show Loading
    loading.style.display = 'block';

    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        });

        if (!response.ok) throw new Error('Network error');

        const data = await response.json();
        
        // 3. UI Update: AI Answer
        appendMessage('ai', data.answer);

    } catch (error) {
        appendMessage('ai', "I'm sorry, I'm having trouble connecting to the medical database right now.");
        console.error(error);
    } finally {
        loading.style.display = 'none';
    }
}

// Event Listeners
sendBtn.addEventListener('click', handleQuery);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleQuery();
});
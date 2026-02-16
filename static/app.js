// Chat interface JavaScript

const chatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');
const chatWindow = document.getElementById('chatWindow');
const loadingIndicator = document.getElementById('loadingIndicator');

let sessionId = generateSessionId();

/**
 * Generate a unique session ID for the chat session
 */
function generateSessionId() {
  return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

/**
 * Add a message to the chat window
 */
function addMessageToChat(content, isUserMessage) {
  const messageElement = document.createElement('div');
  messageElement.className = `message ${isUserMessage ? 'user-message' : 'bot-message'}`;
  
  const contentElement = document.createElement('div');
  contentElement.className = 'message-content';
  contentElement.textContent = content;
  
  messageElement.appendChild(contentElement);
  chatWindow.appendChild(messageElement);
  
  scrollToBottom();
}

/**
 * Auto-scroll chat window to the bottom
 */
function scrollToBottom() {
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

/**
 * Show or hide the loading indicator
 */
function setLoading(isLoading) {
  if (isLoading) {
    loadingIndicator.classList.add('active');
  } else {
    loadingIndicator.classList.remove('active');
  }
}

/**
 * Send a message to the chat API
 */
async function sendMessage() {
  const userMessage = messageInput.value.trim();
  
  if (!userMessage) {
    return;
  }
  
  // Add user message to chat
  addMessageToChat(userMessage, true);
  messageInput.value = '';
  messageInput.focus();
  
  // Show loading indicator
  setLoading(true);
  
  try {
    // Send message to backend
    const response = await fetch('/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message: userMessage,
        session_id: sessionId,
      }),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    
    // Add bot response to chat
    const botMessage = data.message || 'Sorry, I encountered an error processing your request.';
    addMessageToChat(botMessage, false);
    
    // Update session if provided
    if (data.session_id) {
      sessionId = data.session_id;
    }
  } catch (error) {
    console.error('Error sending message:', error);
    addMessageToChat(
      'Sorry, I encountered an error. Please try again later.',
      false
    );
  } finally {
    // Hide loading indicator
    setLoading(false);
  }
}

/**
 * Handle form submission
 */
chatForm.addEventListener('submit', (event) => {
  event.preventDefault();
  sendMessage();
});

/**
 * Allow sending message with Enter key
 */
messageInput.addEventListener('keypress', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});

/**
 * Focus input on page load
 */
document.addEventListener('DOMContentLoaded', () => {
  messageInput.focus();
});

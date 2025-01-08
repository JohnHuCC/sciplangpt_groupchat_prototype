const { useState, useEffect, useRef } = React;

const ChatRoom = () => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState("");
  const [room, setRoom] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const roomId = new URLSearchParams(window.location.search).get("room");

  // Load room data and chat history
  useEffect(() => {
    if (roomId) {
      loadRoom();
      loadMessages();
    }
  }, [roomId]);

  // Auto scroll to bottom
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const loadRoom = async () => {
    try {
      const response = await fetch(`/api/chat/rooms/${roomId}`);
      const data = await response.json();
      if (response.ok) {
        setRoom(data);
      } else {
        setError(data.error || "Failed to load chat room");
      }
    } catch (err) {
      setError("Error loading chat room");
    }
  };

  const loadMessages = async () => {
    try {
      const response = await fetch(`/api/chat/rooms/${roomId}/messages`);
      const data = await response.json();
      if (response.ok) {
        setMessages(data);
      }
    } catch (err) {
      setError("Error loading messages");
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim() || loading) return;

    setLoading(true);
    try {
      const response = await fetch(`/api/chat/rooms/${roomId}/messages`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: inputMessage,
        }),
      });

      const data = await response.json();
      if (response.ok) {
        setMessages((prev) => [
          ...prev,
          data.user_message,
          ...data.agent_responses,
        ]);
        setInputMessage("");
        setError(null);
      } else {
        setError(data.error || "Failed to send message");
      }
    } catch (err) {
      setError("Error sending message");
    } finally {
      setLoading(false);
    }
  };

  if (!room) {
    return <div className="p-6">Loading chat room...</div>;
  }

  return (
    <div className="max-w-6xl mx-auto p-6 h-screen flex flex-col">
      <div className="flex justify-between items-center mb-4">
        <div>
          <h1 className="text-2xl font-bold">{room.name}</h1>
          <div className="text-sm text-gray-500">
            Active Agents: {room.agent_names.join(", ")}
          </div>
        </div>
        <button
          onClick={() => (window.location.href = "/")}
          className="text-blue-600 hover:text-blue-800"
        >
          Back to Agent Pool
        </button>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-100 text-red-700 rounded">{error}</div>
      )}

      <div className="flex-1 overflow-y-auto bg-gray-50 rounded-lg border p-4 mb-4">
        <div className="space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex flex-col ${
                message.sender === "user" ? "items-end" : "items-start"
              }`}
            >
              <div className="flex items-center mb-1">
                <span className="text-sm font-medium text-gray-700">
                  {message.sender}
                </span>
                <span className="text-xs text-gray-500 ml-2">
                  {new Date(message.timestamp).toLocaleTimeString()}
                </span>
              </div>
              <div
                className={`rounded-lg px-4 py-2 max-w-3xl ${
                  message.sender === "user"
                    ? "bg-blue-600 text-white"
                    : "bg-white border"
                }`}
              >
                <p className="whitespace-pre-wrap">{message.content}</p>
                {message.error && (
                  <p className="text-red-500 text-sm mt-1">{message.error}</p>
                )}
                {message.used_knowledge &&
                  message.used_knowledge.length > 0 && (
                    <div className="mt-2 text-sm text-gray-500">
                      <p className="font-medium">Referenced Knowledge:</p>
                      {message.used_knowledge.map((item, index) => (
                        <div key={index} className="ml-4 mt-1">
                          •{" "}
                          {item.text && item.text.length > 100
                            ? `${item.text.substring(0, 100)}...`
                            : item.text}
                          <span className="text-xs ml-1">
                            (Source: {item.source})
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>

      <form onSubmit={handleSendMessage} className="flex space-x-4">
        <input
          type="text"
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          placeholder="Type your message..."
          className="flex-1 p-2 border rounded"
          disabled={loading}
        />
        <button
          type="submit"
          disabled={loading || !inputMessage.trim()}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:bg-gray-300"
        >
          {loading ? "Sending..." : "Send"}
        </button>
      </form>
    </div>
  );
};

ReactDOM.render(<ChatRoom />, document.getElementById("root"));

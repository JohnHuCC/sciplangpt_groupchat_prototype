// ChatRoom.js
const { useState, useEffect, useRef } = React;

const ChatRoom = () => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState("");
  const [room, setRoom] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const messagesEndRef = useRef(null);
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const roomId = new URLSearchParams(window.location.search).get("room");

  // WebSocket 初始化
  const initWebSocket = () => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.hostname;
    const port = window.location.port || (protocol === "wss:" ? "443" : "80");
    const wsUrl = `${protocol}//${host}:${port}/chat/rooms/${roomId}/ws`;

    console.log("Attempting to connect to WebSocket:", wsUrl);
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("WebSocket Connected");
      setWsConnected(true);
      setError(null);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log("Received WebSocket message:", data);
        handleWebSocketMessage(data);
      } catch (err) {
        console.error("Error parsing WebSocket message:", err);
      }
    };

    ws.onclose = (event) => {
      console.log("WebSocket Disconnected", event);
      setWsConnected(false);
      wsRef.current = null;

      if (event.code !== 1000 && event.code !== 1001) {
        reconnectTimeoutRef.current = setTimeout(initWebSocket, 3000);
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket Error:", error);
      setError("Connection error occurred. Retrying...");
    };
  };

  // 處理 WebSocket 消息
  const handleWebSocketMessage = (data) => {
    try {
      console.log("Processing message data:", data);
      if (data.type === "message") {
        const newMessage = data.message;
        setMessages((prevMessages) => {
          // 檢查消息是否為 loading 狀態的更新
          if (newMessage.reply_to) {
            console.log("Updating existing message:", newMessage);
            return prevMessages.map((msg) =>
              msg.id === newMessage.reply_to ? newMessage : msg
            );
          } else {
            console.log("Adding new message:", newMessage);
            return [...prevMessages, newMessage];
          }
        });
      }
    } catch (err) {
      console.error("Error handling WebSocket message:", err);
    }
  };

  // Loading 動畫組件
  const LoadingDots = () => (
    <div className="flex space-x-2 items-center">
      <div className="animate-pulse w-2 h-2 bg-blue-500 rounded-full"></div>
      <div className="animate-pulse w-2 h-2 bg-blue-500 rounded-full animation-delay-200"></div>
      <div className="animate-pulse w-2 h-2 bg-blue-500 rounded-full animation-delay-400"></div>
    </div>
  );

  // 消息內容組件
  const MessageContent = ({ message }) => {
    const isLoading = message.status === "loading";

    if (isLoading) {
      return (
        <div className="flex items-center space-x-2">
          <LoadingDots />
          <span className="text-gray-500">思考中...</span>
        </div>
      );
    }

    return (
      <div>
        <p className="whitespace-pre-wrap break-words">{message.content}</p>
        {message.error && (
          <p className="text-red-500 text-sm mt-1">{message.error}</p>
        )}
        {message.used_knowledge && message.used_knowledge.length > 0 && (
          <div className="mt-2 text-sm text-gray-500">
            <p className="font-medium">Referenced Knowledge:</p>
            <div className="ml-4">
              {message.used_knowledge.map((item, index) => (
                <div key={index} className="mt-1">
                  • {item.text || item.source}
                  {item.source && (
                    <span className="text-xs ml-1">
                      (Source: {item.source})
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  // 消息氣泡組件
  const MessageBubble = ({ message }) => {
    const isUser = message.sender === "user";

    return (
      <div
        className={`flex flex-col ${isUser ? "items-end" : "items-start"} mb-4`}
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
            isUser
              ? "bg-blue-600 text-white"
              : "bg-white border border-gray-200"
          }`}
        >
          <MessageContent message={message} />
        </div>
      </div>
    );
  };

  // 加載房間信息
  const loadRoom = async () => {
    try {
      const response = await fetch(`/api/chat/rooms/${roomId}`);
      if (!response.ok) throw new Error(await response.text());
      const data = await response.json();
      console.log("Room data loaded:", data);
      setRoom(data);
    } catch (err) {
      console.error("Error loading room:", err);
      setError(err.message || "Error loading chat room");
    }
  };

  // 加載消息歷史
  const loadMessages = async () => {
    try {
      const response = await fetch(`/api/chat/rooms/${roomId}/messages`);
      if (!response.ok) throw new Error(await response.text());
      const data = await response.json();
      console.log("Initial messages loaded:", data);
      setMessages(data);
    } catch (err) {
      console.error("Error loading messages:", err);
      setError(err.message || "Error loading messages");
    }
  };

  // 發送消息
  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim() || loading || !wsConnected) return;

    const messageData = {
      type: "chat_message",
      room_id: roomId,
      message: inputMessage,
      timestamp: new Date().toISOString(),
    };

    try {
      setLoading(true);
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        throw new Error("WebSocket not connected");
      }
      console.log("Sending message:", messageData);
      wsRef.current.send(JSON.stringify(messageData));
      setInputMessage("");
      setError(null);
    } catch (err) {
      console.error("Error sending message:", err);
      setError(err.message || "Error sending message");
    } finally {
      setLoading(false);
    }
  };

  // 組件掛載時初始化
  useEffect(() => {
    if (roomId) {
      console.log("Initializing chat room:", roomId);
      initWebSocket();
      loadRoom();
      loadMessages();
    }

    return () => {
      if (wsRef.current) wsRef.current.close(1000);
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [roomId]);

  // 自動滾動到底部
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  if (!room) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="text-xl mb-4">Loading chat room...</div>
          {error && <div className="text-red-500">{error}</div>}
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto p-6 h-screen flex flex-col">
      <div className="flex justify-between items-center mb-4">
        <div>
          <h1 className="text-2xl font-bold">{room.name}</h1>
          <div className="text-sm text-gray-500">
            Active Agents: {room.agent_names.join(", ")}
          </div>
          <div
            className={`text-sm ${
              wsConnected ? "text-green-500" : "text-red-500"
            }`}
          >
            {wsConnected ? "Connected" : "Disconnected"}
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
        <div className="mb-4 p-3 bg-red-100 text-red-700 rounded">
          {error}
          <button
            onClick={() => setError(null)}
            className="ml-2 text-sm hover:text-red-900"
          >
            Dismiss
          </button>
        </div>
      )}

      <div className="flex-1 overflow-y-auto bg-gray-50 rounded-lg border p-4 mb-4">
        {messages.length === 0 ? (
          <div className="text-center text-gray-500">No messages yet</div>
        ) : (
          <div className="space-y-4">
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSendMessage} className="flex space-x-4">
        <input
          type="text"
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          placeholder={wsConnected ? "Type your message..." : "Connecting..."}
          className="flex-1 p-2 border rounded"
          disabled={loading || !wsConnected}
        />
        <button
          type="submit"
          disabled={loading || !wsConnected || !inputMessage.trim()}
          className={`px-4 py-2 rounded ${
            loading || !wsConnected || !inputMessage.trim()
              ? "bg-gray-300 cursor-not-allowed"
              : "bg-blue-600 text-white hover:bg-blue-700"
          }`}
        >
          {loading ? "Sending..." : "Send"}
        </button>
      </form>
    </div>
  );
};

// 渲染應用
ReactDOM.render(<ChatRoom />, document.getElementById("root"));

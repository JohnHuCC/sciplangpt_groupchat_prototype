// app.js
const { useState, useEffect } = React;

const App = () => {
  const [agents, setAgents] = useState([]);
  const [selectedAgents, setSelectedAgents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isCreating, setIsCreating] = useState(false);
  const [newAgent, setNewAgent] = useState({
    name: "",
    description: "",
    template_name: "research_assistant",
    files: [],
  });

  // Load agents when component mounts
  useEffect(() => {
    let mounted = true;

    const loadAgents = async () => {
      try {
        setLoading(true);
        const response = await fetch("/api/agents");

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`Failed to load agents: ${errorText}`);
        }

        const data = await response.json();
        console.log("Received agents data:", data);

        if (mounted) {
          setAgents(Array.isArray(data) ? data : []);
          setError(null);
        }
      } catch (err) {
        console.error("Error loading agents:", err);
        if (mounted) {
          setError(err.message);
          setAgents([]);
        }
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    };

    loadAgents();

    return () => {
      mounted = false;
    };
  }, []);

  // Create new agent
  const handleCreateAgent = async (e) => {
    e.preventDefault();
    try {
      setLoading(true);
      setError(null);

      const formData = new FormData();
      formData.append("name", newAgent.name);
      formData.append("description", newAgent.description);
      formData.append("template_name", newAgent.template_name);

      if (newAgent.files.length > 0) {
        for (let file of newAgent.files) {
          formData.append("files", file);
        }
      }

      const response = await fetch("/api/agents", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText);
      }

      const data = await response.json();
      setAgents((prev) => [...prev, data]);
      setIsCreating(false);
      setNewAgent({
        name: "",
        description: "",
        template_name: "research_assistant",
        files: [],
      });
    } catch (err) {
      console.error("Error creating agent:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Delete agent
  const handleDeleteAgent = async (name) => {
    if (!confirm(`Are you sure you want to delete agent "${name}"?`)) {
      return;
    }

    try {
      const response = await fetch(`/api/agents/${name}`, {
        method: "DELETE",
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText);
      }

      setAgents((prev) => prev.filter((a) => a.name !== name));
      setSelectedAgents((prev) => prev.filter((a) => a.name !== name));
    } catch (err) {
      console.error("Error deleting agent:", err);
      setError(err.message);
    }
  };

  // Handle agent selection
  const handleAgentSelect = (agent) => {
    setSelectedAgents((prev) => {
      const isSelected = prev.find((a) => a.name === agent.name);
      if (isSelected) {
        return prev.filter((a) => a.name !== agent.name);
      }
      return [...prev, agent];
    });
  };

  // Start chat
  const startChat = async () => {
    if (selectedAgents.length === 0) {
      setError("Please select at least one agent");
      return;
    }

    try {
      const response = await fetch("/api/chat/rooms", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name: `Chat Room ${Date.now()}`,
          agent_names: selectedAgents.map((a) => a.name),
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText);
      }

      const data = await response.json();
      window.location.href = `/chat?room=${data.id}`;
    } catch (err) {
      console.error("Error creating chat room:", err);
      setError(err.message);
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Agent Pool</h1>
        <div className="space-x-4">
          <button
            onClick={() => setIsCreating(true)}
            className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700"
            disabled={loading}
          >
            Create New Agent
          </button>
          <button
            onClick={startChat}
            disabled={selectedAgents.length === 0 || loading}
            className={`px-4 py-2 rounded ${
              selectedAgents.length > 0 && !loading
                ? "bg-blue-600 text-white hover:bg-blue-700"
                : "bg-gray-300 text-gray-500 cursor-not-allowed"
            }`}
          >
            Start Chat ({selectedAgents.length} selected)
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-6 p-4 bg-red-100 text-red-700 rounded flex justify-between items-center">
          <span>{error}</span>
          <button
            onClick={() => setError(null)}
            className="text-red-500 hover:text-red-700"
          >
            ✕
          </button>
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="text-center py-8">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-blue-500 border-t-transparent"></div>
          <p className="mt-2 text-gray-600">Loading...</p>
        </div>
      )}

      {/* Agents Grid */}
      {!loading && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {agents.length > 0 ? (
            agents.map((agent) => (
              <div
                key={agent.name}
                className={`border rounded-lg p-6 transition-all cursor-pointer ${
                  selectedAgents.find((a) => a.name === agent.name)
                    ? "border-blue-500 shadow-lg bg-blue-50"
                    : "hover:shadow-lg"
                }`}
                onClick={() => handleAgentSelect(agent)}
              >
                <h3 className="text-xl font-semibold mb-2">{agent.name}</h3>
                <p className="text-gray-600 mb-4">{agent.description}</p>
                <div className="text-sm text-gray-500 mb-4">
                  Created: {new Date(agent.created_at).toLocaleDateString()}
                </div>
                <div className="flex justify-end space-x-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteAgent(agent.name);
                    }}
                    className="px-3 py-1 text-sm bg-red-100 text-red-700 rounded hover:bg-red-200"
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))
          ) : (
            <div className="col-span-3 text-center py-8 text-gray-500">
              No agents available
            </div>
          )}
        </div>
      )}

      {/* Create Agent Modal */}
      {isCreating && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg max-w-2xl w-full p-6">
            <h2 className="text-xl font-bold mb-4">Create New Agent</h2>
            <form onSubmit={handleCreateAgent} className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">Name</label>
                <input
                  type="text"
                  value={newAgent.name}
                  onChange={(e) =>
                    setNewAgent({ ...newAgent, name: e.target.value })
                  }
                  className="w-full p-2 border rounded"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">
                  Description
                </label>
                <textarea
                  value={newAgent.description}
                  onChange={(e) =>
                    setNewAgent({ ...newAgent, description: e.target.value })
                  }
                  className="w-full p-2 border rounded h-24"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">
                  Template
                </label>
                <select
                  value={newAgent.template_name}
                  onChange={(e) =>
                    setNewAgent({ ...newAgent, template_name: e.target.value })
                  }
                  className="w-full p-2 border rounded"
                  required
                >
                  <option value="research_assistant">Research Assistant</option>
                  <option value="general_qa">QA Agent</option>
                  <option value="content_writer">Content Writer</option>
                  <option value="market_analyst">Market Analyst</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">
                  Knowledge Files (Optional)
                </label>
                <input
                  type="file"
                  multiple
                  onChange={(e) =>
                    setNewAgent({
                      ...newAgent,
                      files: Array.from(e.target.files || []),
                    })
                  }
                  className="w-full p-2 border rounded"
                  accept=".txt,.pdf,.doc,.docx"
                />
              </div>

              <div className="flex justify-end space-x-2 mt-6">
                <button
                  type="button"
                  onClick={() => setIsCreating(false)}
                  className="px-4 py-2 border rounded hover:bg-gray-100"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={loading}
                  className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:bg-gray-400"
                >
                  {loading ? "Creating..." : "Create Agent"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

// 只在根元素存在時進行渲染
const rootElement = document.getElementById("root");
if (rootElement) {
  ReactDOM.render(<App />, rootElement);
}

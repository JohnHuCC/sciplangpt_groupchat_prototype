// AgentPoolApp.js
const AgentPool = () => {
  // State hooks
  const [agents, setAgents] = React.useState([]);
  const [selectedAgents, setSelectedAgents] = React.useState([]);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState(null);
  const [isCreating, setIsCreating] = React.useState(false);
  const [newAgent, setNewAgent] = React.useState({
    name: "",
    description: "",
    files: [],
    base_prompt: "",
    query_templates: [""],
  });

  // Effect for loading agents
  React.useEffect(() => {
    loadAgents();
  }, []);

  // Load agents list
  const loadAgents = async () => {
    try {
      const response = await fetch("/api/agents");
      const data = await response.json();
      setAgents(data);
    } catch (err) {
      setError("Failed to load agents");
      console.error("Error:", err);
    }
  };

  // Handle agent creation
  const handleCreateAgent = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("name", newAgent.name);
    formData.append("description", newAgent.description);
    formData.append("base_prompt", newAgent.base_prompt);

    // Append files
    for (let file of newAgent.files) {
      formData.append("files[]", file);
    }

    // Append query templates
    newAgent.query_templates.forEach((template, index) => {
      formData.append(`query_template_${index}`, template);
    });

    try {
      const response = await fetch("/api/agents", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (response.ok) {
        await loadAgents();
        setIsCreating(false);
        setNewAgent({
          name: "",
          description: "",
          files: [],
          base_prompt: "",
          query_templates: [""],
        });
      } else {
        setError(data.error || "Failed to create agent");
      }
    } catch (err) {
      setError("Error creating agent");
      console.error("Error:", err);
    } finally {
      setLoading(false);
    }
  };

  // Handle agent deletion
  const handleDeleteAgent = async (name) => {
    if (!confirm(`Are you sure you want to delete agent "${name}"?`)) {
      return;
    }

    try {
      const response = await fetch(`/api/agents/${name}`, {
        method: "DELETE",
      });

      if (response.ok) {
        await loadAgents();
        // Remove from selected agents if it was selected
        setSelectedAgents((prev) => prev.filter((a) => a.name !== name));
      } else {
        const data = await response.json();
        setError(data.error || "Failed to delete agent");
      }
    } catch (err) {
      setError("Error deleting agent");
      console.error("Error:", err);
    }
  };

  const handleAgentSelect = (agent) => {
    if (selectedAgents.find((a) => a.name === agent.name)) {
      setSelectedAgents((prev) => prev.filter((a) => a.name !== agent.name));
    } else {
      setSelectedAgents((prev) => [...prev, agent]);
    }
  };

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
          agent_names: selectedAgents.map((agent) => agent.name),
        }),
      });

      const data = await response.json();
      if (response.ok) {
        window.location.href = `/chat?room=${data.id}`;
      } else {
        setError(data.error || "Failed to create chat room");
      }
    } catch (err) {
      setError("Error creating chat room");
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
          >
            Create New Agent
          </button>
          <button
            onClick={startChat}
            disabled={selectedAgents.length === 0}
            className={`px-4 py-2 rounded ${
              selectedAgents.length > 0
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
        <div className="mb-6 p-4 bg-red-100 text-red-700 rounded">{error}</div>
      )}

      {/* Agents Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {agents.map((agent) => (
          <div
            key={agent.name}
            className={`border rounded-lg p-6 transition-all cursor-pointer
              ${
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
        ))}
      </div>

      {/* Create Agent Form Modal */}
      {isCreating && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg max-w-2xl w-full p-6">
            <h2 className="text-xl font-bold mb-4">Create New Agent</h2>
            <form onSubmit={handleCreateAgent} className="space-y-4">
              {/* Name Input */}
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

              {/* Description Input */}
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

              {/* File Upload */}
              <div>
                <label className="block text-sm font-medium mb-1">
                  Knowledge Files
                </label>
                <input
                  type="file"
                  multiple
                  onChange={(e) =>
                    setNewAgent({
                      ...newAgent,
                      files: Array.from(e.target.files),
                    })
                  }
                  className="w-full p-2 border rounded"
                  accept=".txt,.pdf,.doc,.docx"
                  required
                />
              </div>

              {/* Base Prompt */}
              <div>
                <label className="block text-sm font-medium mb-1">
                  Base Prompt
                </label>
                <textarea
                  value={newAgent.base_prompt}
                  onChange={(e) =>
                    setNewAgent({ ...newAgent, base_prompt: e.target.value })
                  }
                  className="w-full p-2 border rounded h-32"
                  placeholder="Enter the base prompt for this agent..."
                  required
                />
              </div>

              {/* Query Templates */}
              <div>
                <label className="block text-sm font-medium mb-1">
                  Query Templates
                  <button
                    type="button"
                    onClick={() =>
                      setNewAgent({
                        ...newAgent,
                        query_templates: [...newAgent.query_templates, ""],
                      })
                    }
                    className="ml-2 px-2 py-1 text-sm bg-gray-100 rounded hover:bg-gray-200"
                  >
                    + Add Template
                  </button>
                </label>
                {newAgent.query_templates.map((template, index) => (
                  <div key={index} className="flex mb-2">
                    <textarea
                      value={template}
                      onChange={(e) => {
                        const newTemplates = [...newAgent.query_templates];
                        newTemplates[index] = e.target.value;
                        setNewAgent({
                          ...newAgent,
                          query_templates: newTemplates,
                        });
                      }}
                      className="flex-1 p-2 border rounded"
                      placeholder={`Query template ${index + 1}`}
                    />
                    {newAgent.query_templates.length > 1 && (
                      <button
                        type="button"
                        onClick={() => {
                          const newTemplates = newAgent.query_templates.filter(
                            (_, i) => i !== index
                          );
                          setNewAgent({
                            ...newAgent,
                            query_templates: newTemplates,
                          });
                        }}
                        className="ml-2 px-2 text-red-600 hover:text-red-800"
                      >
                        ×
                      </button>
                    )}
                  </div>
                ))}
              </div>

              {/* Form Buttons */}
              <div className="flex justify-end space-x-2">
                <button
                  type="button"
                  onClick={() => setIsCreating(false)}
                  className="px-4 py-2 border rounded hover:bg-gray-100"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
                  disabled={loading}
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

// Render the app
ReactDOM.render(<AgentPool />, document.getElementById("root"));

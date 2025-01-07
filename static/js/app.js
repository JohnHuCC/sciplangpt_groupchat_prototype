const App = () => {
  const [selectedFiles, setSelectedFiles] = React.useState([]);
  const [agents, setAgents] = React.useState([]);
  const [selectedAgent, setSelectedAgent] = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState(null);
  const [researchArea, setResearchArea] = React.useState("");
  const [results, setResults] = React.useState(null);
  const [isUploading, setIsUploading] = React.useState(false);
  const [uploadStatus, setUploadStatus] = React.useState("");
  const [stats, setStats] = React.useState({
    file_count: 0,
    total_chunks: 0,
    total_size: "0 B",
  });
  // 在組件加載時獲取現有 agents
  React.useEffect(() => {
    fetchAgents();
  }, []);

  const fetchAgents = async () => {
    try {
      const response = await fetch("/api/agents");
      const data = await response.json();
      setAgents(data.agents);
    } catch (err) {
      console.error("Error fetching agents:", err);
    }
  };

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files);
    setSelectedFiles((prev) => [...prev, ...files]);
  };

  const handleFileUpload = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    const files = Array.from(
      e.target.querySelector('input[type="file"]').files
    );

    if (files.length === 0) {
      setError("Please select at least one file");
      return;
    }

    setIsUploading(true);
    setUploadStatus("Uploading files...");
    setError(null);

    // 添加所有檔案到 FormData
    files.forEach((file) => {
      formData.append("files[]", file);
    });

    try {
      const response = await fetch("/upload_to_knowledge", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.error) {
        setError(data.error);
        setUploadStatus("");
      } else {
        setUploadStatus(
          `Successfully uploaded ${data.uploaded_files.length} files`
        );
        // 顯示成功和錯誤的詳細信息
        if (data.errors && data.errors.length > 0) {
          setUploadStatus((prev) => prev + "\nSome files had errors:");
          data.errors.forEach((error) => {
            setUploadStatus((prev) => prev + `\n${error.name}: ${error.error}`);
          });
        }

        // 更新統計信息
        const statsResponse = await fetch("/api/knowledge_base_stats");
        const statsData = await statsResponse.json();
        setStats(statsData);

        // 3秒後清除上傳狀態
        setTimeout(() => {
          setUploadStatus("");
          // 清除檔案選擇
          e.target.reset();
        }, 3000);
      }
    } catch (err) {
      setError(err.message);
      setUploadStatus("");
    } finally {
      setIsUploading(false);
    }
  };

  const handleCreateAgent = async () => {
    if (selectedFiles.length === 0) {
      setError("Please select files first");
      return;
    }

    setLoading(true);
    setError(null);
    setUploadStatus("Creating agent..."); // 添加上傳狀態

    const formData = new FormData();
    selectedFiles.forEach((file) => {
      formData.append("files[]", file); // 改為 files[]
    });

    try {
      const response = await fetch("/api/create_agent", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (data.error) {
        setError(data.error);
      } else {
        setUploadStatus(
          `Agent created successfully with ${data.uploaded_files.length} files`
        );
        if (data.errors) {
          setUploadStatus((prev) => prev + "\nSome files had errors:");
          data.errors.forEach((error) => {
            setUploadStatus((prev) => prev + `\n${error.name}: ${error.error}`);
          });
        }
        // 重新獲取 agents 列表
        await fetchAgents();
        setSelectedFiles([]); // 清空已選文件

        // 3秒後清除上傳狀態
        setTimeout(() => {
          setUploadStatus("");
        }, 3000);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleGeneratePlan = async () => {
    if (!selectedAgent) {
      setError("Please select an agent first");
      return;
    }
    if (!researchArea.trim()) {
      setError("Please enter a research area");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      console.log("Sending request with:", {
        agent_id: selectedAgent,
        research_area: researchArea,
      });

      const response = await fetch("/api/generate_plan", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          agent_id: selectedAgent,
          research_area: researchArea,
        }),
      });

      if (!response.ok) {
        const text = await response.text();
        console.error("Server response:", text);
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      if (data.error) {
        setError(data.error);
      } else {
        setResults(data);
      }
    } catch (err) {
      console.error("Error details:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-8">Research Agent System</h1>

      {/* File Upload and Agent Creation Section */}
      <div className="mb-8 p-6 border rounded-lg bg-gray-50">
        <h2 className="text-xl font-semibold mb-4">Create New Agent</h2>

        {/* File Selection */}
        <div className="mb-4">
          <input
            type="file"
            multiple
            onChange={handleFileSelect}
            className="mb-2"
          />
          <div className="text-sm text-gray-600">
            Selected files: {selectedFiles.map((f) => f.name).join(", ")}
          </div>
        </div>

        {/* Create Agent Button */}
        <button
          onClick={handleCreateAgent}
          disabled={loading || selectedFiles.length === 0}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:bg-blue-300"
        >
          {loading ? "Creating Agent..." : "Create Agent"}
        </button>
      </div>

      {/* Agent Selection and Research Plan Generation */}
      <div className="mb-8 p-6 border rounded-lg bg-gray-50">
        <h2 className="text-xl font-semibold mb-4">Generate Research Plan</h2>

        {/* Agent Selection */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">Select Agent</label>
          <select
            value={selectedAgent || ""}
            onChange={(e) => setSelectedAgent(e.target.value)}
            className="w-full p-2 border rounded"
          >
            <option value="">-- Select an Agent --</option>
            {agents.map((agent) => (
              <option key={agent.id} value={agent.id}>
                {agent.name}
              </option>
            ))}
          </select>
        </div>

        {/* Research Area Input */}
        <div className="mb-4">
          <input
            type="text"
            value={researchArea}
            onChange={(e) => setResearchArea(e.target.value)}
            placeholder="Enter research area"
            className="w-full p-3 border rounded"
          />
        </div>

        {/* Generate Plan Button */}
        <button
          onClick={handleGeneratePlan}
          disabled={loading || !selectedAgent || !researchArea.trim()}
          className="w-full bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 disabled:bg-green-300"
        >
          {loading ? "Generating..." : "Generate Research Plan"}
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-6 p-4 bg-red-100 text-red-700 rounded-md">
          <p>Error: {error}</p>
        </div>
      )}

      {/* Results Display */}
      {results && (
        <div className="space-y-6">
          {/* Display results similarly to before */}
        </div>
      )}
    </div>
  );
};

ReactDOM.render(<App />, document.getElementById("root"));

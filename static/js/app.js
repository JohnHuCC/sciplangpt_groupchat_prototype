const App = () => {
  const [loading, setLoading] = React.useState(false);
  const [results, setResults] = React.useState(null);
  const [error, setError] = React.useState(null);
  const [editedPrompt, setEditedPrompt] = React.useState("");
  const [showEditPrompt, setShowEditPrompt] = React.useState(false);
  const [generationStatus, setGenerationStatus] = React.useState("");
  const [researchArea, setResearchArea] = React.useState("");
  const [isUploading, setIsUploading] = React.useState(false);
  const [uploadStatus, setUploadStatus] = React.useState("");
  const [stats, setStats] = React.useState({
    file_count: 0,
    total_chunks: 0,
    total_size: "0 B",
  });

  // 在組件加載時獲取統計信息
  React.useEffect(() => {
    fetch("/api/knowledge_base_stats")
      .then((response) => response.json())
      .then((data) => setStats(data))
      .catch((err) => console.error("Error loading stats:", err));
  }, []);

  const handleFileUpload = async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    setIsUploading(true);
    setUploadStatus("Uploading file to knowledge base...");
    setError(null);

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
        setUploadStatus("File successfully added to knowledge base!");
        // 更新統計信息
        const statsResponse = await fetch("/api/knowledge_base_stats");
        const statsData = await statsResponse.json();
        setStats(statsData);
        setTimeout(() => setUploadStatus(""), 3000);
      }
    } catch (err) {
      setError(err.message);
      setUploadStatus("");
    } finally {
      setIsUploading(false);
    }
  };

  const handleGeneratePlan = async (e) => {
    e.preventDefault();
    if (!researchArea.trim()) {
      setError("Please enter a research area");
      return;
    }

    setLoading(true);
    setError(null);
    setGenerationStatus("Starting generation process...");

    try {
      const response = await fetch("/generate_plan", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          research_area: researchArea,
        }),
      });

      const data = await response.json();

      if (data.error) {
        setError(data.error);
      } else {
        setResults(data);
        setEditedPrompt(data.research_prompt);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      setGenerationStatus("");
    }
  };

  const handlePromptEdit = async () => {
    setLoading(true);
    setGenerationStatus("Generating new plan...");

    try {
      const response = await fetch("/generate_with_prompt", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          research_prompt: editedPrompt,
        }),
      });

      const data = await response.json();

      if (data.error) {
        setError(data.error);
      } else {
        setResults({
          ...results,
          research_prompt: editedPrompt,
          research_plan: data.research_plan,
        });
        setShowEditPrompt(false);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      setGenerationStatus("");
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-8">Research Plan Generator</h1>

      {/* File Upload Section */}
      <div className="mb-12 p-6 border rounded-lg bg-gray-50">
        <h2 className="text-xl font-semibold mb-4">1. Build Knowledge Base</h2>
        <form onSubmit={handleFileUpload} className="space-y-4">
          <div className="border-2 border-dashed rounded-lg p-8 text-center">
            <input
              type="file"
              name="file"
              accept=".pdf,.txt,.docx"
              required
              className="w-full"
              disabled={isUploading}
            />
            <p className="mt-2 text-sm text-gray-600">
              Support PDF, TXT, or DOCX files
            </p>
          </div>
          <button
            type="submit"
            className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:bg-blue-300"
            disabled={isUploading}
          >
            {isUploading ? "Processing..." : "Upload to Knowledge Base"}
          </button>
        </form>
        {uploadStatus && (
          <div className="mt-4 p-3 bg-blue-50 text-blue-700 rounded-md flex items-center">
            {isUploading && (
              <div className="w-5 h-5 mr-3">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-700"></div>
              </div>
            )}
            <p>{uploadStatus}</p>
          </div>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-6 p-4 bg-red-100 text-red-700 rounded-md">
          <p>Error: {error}</p>
        </div>
      )}

      {/* Research Plan Generation Section */}
      <div className="mb-12 p-6 border rounded-lg bg-gray-50">
        <h2 className="text-xl font-semibold mb-4">
          2. Generate Research Plan
        </h2>
        <form onSubmit={handleGeneratePlan} className="space-y-4">
          <div>
            <input
              type="text"
              value={researchArea}
              onChange={(e) => setResearchArea(e.target.value)}
              placeholder="Enter research area (e.g., Artificial Intelligence in Healthcare)"
              className="w-full p-3 border rounded"
              required
            />
          </div>
          <button
            type="submit"
            className="w-full bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 disabled:bg-green-300"
            disabled={loading}
          >
            {loading ? "Generating..." : "Generate Research Plan"}
          </button>
        </form>

        {/* Generation Status Indicator */}
        {generationStatus && (
          <div className="mt-4 p-3 bg-blue-50 text-blue-700 rounded-md flex items-center">
            <div className="w-5 h-5 mr-3">
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-700"></div>
            </div>
            <p>{generationStatus}</p>
          </div>
        )}
      </div>

      {/* Results Section */}
      {results && (
        <div className="space-y-6">
          {/* Knowledge Base Stats */}
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <h2 className="text-xl font-semibold mb-4">
              Knowledge Base Information
            </h2>
            <div className="grid grid-cols-3 gap-4 mb-4">
              <div className="p-4 bg-gray-50 rounded">
                <div className="text-sm text-gray-500">Total Files</div>
                <div className="text-xl font-semibold">{stats.file_count}</div>
              </div>
              <div className="p-4 bg-gray-50 rounded">
                <div className="text-sm text-gray-500">Total Chunks</div>
                <div className="text-xl font-semibold">
                  {stats.total_chunks}
                </div>
              </div>
              <div className="p-4 bg-gray-50 rounded">
                <div className="text-sm text-gray-500">Total Size</div>
                <div className="text-xl font-semibold">{stats.total_size}</div>
              </div>
            </div>
            <a
              href="/view_knowledge_base"
              className="text-blue-600 hover:text-blue-800 flex items-center"
            >
              View Knowledge Base Contents →
            </a>
          </div>

          {/* Used Knowledge Section */}
          {results.used_knowledge && (
            <div className="bg-white p-6 rounded-lg shadow-sm border">
              <h2 className="text-xl font-semibold mb-4">
                Referenced Knowledge
              </h2>
              <div className="space-y-4">
                {results.used_knowledge.map((item, index) => (
                  <div key={index} className="p-4 bg-gray-50 rounded">
                    <div className="text-sm text-gray-500 mb-2">
                      Source: {item.source}
                    </div>
                    <div className="text-gray-700">{item.text}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Step 1: Research Area */}
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <h2 className="text-xl font-semibold mb-2">
              Step 1: Research Area Detection
            </h2>
            <div className="bg-gray-50 p-4 rounded">
              <p>{results.research_area}</p>
            </div>
          </div>

          {/* Step 2: Research Question */}
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <h2 className="text-xl font-semibold mb-2">
              Step 2: Generated Research Question
            </h2>
            <div className="bg-gray-50 p-4 rounded">
              <pre className="whitespace-pre-wrap">
                {results.research_question}
              </pre>
            </div>
          </div>

          {/* Step 3: Research Prompt */}
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <h2 className="text-xl font-semibold mb-2">
              Step 3: Research Prompt
              <button
                onClick={() => setShowEditPrompt(!showEditPrompt)}
                className="ml-4 px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
              >
                {showEditPrompt ? "Cancel Edit" : "Edit Prompt"}
              </button>
            </h2>
            {showEditPrompt ? (
              <div className="space-y-4">
                <textarea
                  value={editedPrompt}
                  onChange={(e) => setEditedPrompt(e.target.value)}
                  className="w-full h-64 p-4 border rounded"
                />
                <button
                  onClick={handlePromptEdit}
                  className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
                  disabled={loading}
                >
                  Generate New Plan with Edited Prompt
                </button>
              </div>
            ) : (
              <div className="bg-gray-50 p-4 rounded">
                <pre className="whitespace-pre-wrap">
                  {results.research_prompt}
                </pre>
              </div>
            )}
          </div>

          {/* Step 4: Final Research Plan with Status */}
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <h2 className="text-xl font-semibold mb-2">
              Step 4: Final Research Plan
            </h2>

            {/* Generation Status Indicator */}
            {generationStatus && (
              <div className="mb-4 p-3 bg-blue-50 text-blue-700 rounded-md flex items-center">
                <div className="w-5 h-5 mr-3">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-700"></div>
                </div>
                <p>{generationStatus}</p>
              </div>
            )}

            <div className="bg-gray-50 p-4 rounded">
              {loading ? (
                <div className="text-center p-4">
                  <div className="animate-pulse space-y-4">
                    <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                    <div className="h-4 bg-gray-200 rounded"></div>
                    <div className="h-4 bg-gray-200 rounded w-5/6"></div>
                  </div>
                </div>
              ) : (
                <pre className="whitespace-pre-wrap">
                  {results.research_plan}
                </pre>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

ReactDOM.render(<App />, document.getElementById("root"));

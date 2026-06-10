"use client";

import React, { useState, useEffect, useRef } from "react";
import styles from "./page.module.css";

// Interface definitions for visualization tree
interface RLMNode {
  id: string;
  type: "engine" | "orchestrator" | "executor" | "leaf_call" | "branch_call";
  depth: number;
  label: string;
  query?: string;
  contextLen?: number;
  content?: string;
  finalAnswer?: string;
  variablesSummary?: string;
  code?: string | null;
  stdout?: string;
  stderr?: string;
  success?: boolean;
  exception?: string;
  textSliceLen?: number;
  response?: string;
  children: RLMNode[];
  timestamp: number;
}

export default function Dashboard() {
  // Form State
  const [query, setQuery] = useState("Identify major projects and their budgets mentioned in the report.");
  const [provider, setProvider] = useState("mock");
  const [modelName, setModelName] = useState("gpt-4o-mini");
  const [maxDepth, setMaxDepth] = useState(3);
  const [maxSteps, setMaxSteps] = useState(10);
  const [contextSource, setContextSource] = useState<"text" | "file">("text");
  const [contextText, setContextText] = useState(
    "Annual Operations Summary Report:\n\n" +
    "Project Apollo was approved in March. The engineering team has completed phase 1. " +
    "The initial financial allocation is $5,000,000 for server deployment and testing.\n\n" +
    "Project Titan is focused on database modernization. The budget is $12,000,000. " +
    "Milestones include cloud migration and containerization."
  );
  const [file, setFile] = useState<File | null>(null);

  // Execution State
  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState<"idle" | "running" | "success" | "error">("idle");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [events, setEvents] = useState<any[]>([]);
  const [finalAnswer, setFinalAnswer] = useState<string | null>(null);
  
  // Metrics
  const [elapsedTime, setElapsedTime] = useState(0);
  const [characterCount, setCharacterCount] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);
  
  // UI States
  const [collapsedMap, setCollapsedMap] = useState<{ [id: string]: boolean }>({});
  const [viewMode, setViewMode] = useState<"tree" | "graph">("tree");
  const [selectedNode, setSelectedNode] = useState<RLMNode | null>(null);
  
  // Refs for tracking
  const abortControllerRef = useRef<AbortController | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Calculate Character Count
  useEffect(() => {
    if (contextSource === "text") {
      setCharacterCount(contextText.length);
    } else if (file) {
      setCharacterCount(file.size); // approximation
    } else {
      setCharacterCount(0);
    }
  }, [contextText, file, contextSource]);

  // Handle run execution
  const handleRun = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isRunning) return;

    // Reset States
    setIsRunning(true);
    setStatus("running");
    setErrorMsg(null);
    setEvents([]);
    setFinalAnswer(null);
    setElapsedTime(0);
    setTotalSteps(0);
    setCollapsedMap({});

    // Start Timer
    const startTime = Date.now();
    timerRef.current = setInterval(() => {
      setElapsedTime(Math.round((Date.now() - startTime) / 1000));
    }, 1000);

    // Setup Abort Controller
    abortControllerRef.current = new AbortController();

    // Prepare Request Body
    const formData = new FormData();
    formData.append("query", query);
    formData.append("provider", provider);
    formData.append("model_name", modelName);
    formData.append("max_depth", maxDepth.toString());
    formData.append("max_steps", maxSteps.toString());

    if (contextSource === "file" && file) {
      formData.append("file", file);
    } else {
      formData.append("context_text", contextText);
    }

    try {
      const response = await fetch("http://127.0.0.1:8000/api/run", {
        method: "POST",
        body: formData,
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Server returned status ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder("utf-8");

      if (!reader) {
        throw new Error("Response body is not readable.");
      }

      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const event = JSON.parse(line.slice(6));
              
              // Increment step counter on orchestrator decision
              if (event.type === "orchestrator") {
                setTotalSteps((prev) => prev + 1);
              }
              
              setEvents((prev) => [...prev, event]);

              // Handle termination events
              if (event.type === "complete") {
                setFinalAnswer(event.final_answer);
                setStatus("success");
              } else if (event.type === "error") {
                setErrorMsg(event.message);
                setStatus("error");
              }
            } catch (err) {
              console.error("Error parsing event JSON", err);
            }
          }
        }
      }
    } catch (err: any) {
      if (err.name === "AbortError") {
        setStatus("idle");
      } else {
        setErrorMsg(err.message || "An unexpected error occurred.");
        setStatus("error");
      }
    } finally {
      setIsRunning(false);
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    }
  };

  // Stop Execution
  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setIsRunning(false);
    if (timerRef.current) {
      clearInterval(timerRef.current);
    }
  };

  // Clear visualizer
  const handleClear = () => {
    setEvents([]);
    setFinalAnswer(null);
    setErrorMsg(null);
    setStatus("idle");
    setElapsedTime(0);
    setTotalSteps(0);
  };

  // Toggle Collapse Map
  const toggleCollapse = (id: string) => {
    setCollapsedMap((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  // Helper to split Orchestrator markdown
  const parseOrchestratorContent = (content: string) => {
    const codePattern = /```python\s*([\s\S]*?)\s*```/;
    const match = content.match(codePattern);
    const code = match ? match[1].trim() : null;
    const description = content.replace(codePattern, "").trim();
    return { description, code };
  };

  // Construct Tree from Flat events
  const buildTree = (events: any[]): RLMNode[] => {
    const rootNodes: RLMNode[] = [];
    const activeEngines: { [depth: number]: RLMNode } = {};
    const activeBranchCalls: { [depth: number]: RLMNode } = {};

    for (let i = 0; i < events.length; i++) {
      const e = events[i];
      const id = `${e.type}-${i}`;

      if (e.type === "engine_start") {
        const node: RLMNode = {
          id,
          type: "engine",
          depth: e.depth,
          label: `RLM Engine (Depth ${e.depth})`,
          query: e.query,
          contextLen: e.context_len,
          children: [],
          timestamp: Date.now(),
        };

        activeEngines[e.depth] = node;

        if (e.depth === 0) {
          rootNodes.push(node);
        } else {
          const parentBranch = activeBranchCalls[e.depth - 1];
          if (parentBranch) {
            parentBranch.children.push(node);
          } else {
            const parentEngine = activeEngines[e.depth - 1];
            if (parentEngine) {
              parentEngine.children.push(node);
            } else {
              rootNodes.push(node);
            }
          }
        }
      } else if (e.type === "engine_end") {
        const node = activeEngines[e.depth];
        if (node) {
          node.finalAnswer = e.final_answer || undefined;
          node.success = e.success;
        }
      } else {
        const activeEngine = activeEngines[e.depth];
        if (activeEngine) {
          if (e.type === "orchestrator") {
            activeEngine.children.push({
              id,
              type: "orchestrator",
              depth: e.depth,
              label: "Neural Orchestrator Thinking",
              content: e.content,
              finalAnswer: e.final_answer || undefined,
              variablesSummary: e.variables_summary,
              children: [],
              timestamp: Date.now(),
            });
          } else if (e.type === "executor") {
            activeEngine.children.push({
              id,
              type: "executor",
              depth: e.depth,
              label: "Python Executor Code Run",
              code: e.code,
              stdout: e.stdout,
              stderr: e.stderr,
              success: e.success,
              exception: e.exception,
              children: [],
              timestamp: Date.now(),
            });
          } else if (e.type === "leaf_start") {
            activeEngine.children.push({
              id,
              type: "leaf_call",
              depth: e.depth,
              label: "Leaf Query (llm_query)",
              query: e.query,
              textSliceLen: e.text_slice_len,
              children: [],
              timestamp: Date.now(),
            });
          } else if (e.type === "leaf_end") {
            const lastLeaf = [...activeEngine.children]
              .reverse()
              .find((c) => c.type === "leaf_call" && c.query === e.query);
            if (lastLeaf) {
              lastLeaf.response = e.response;
            }
          } else if (e.type === "branch_start") {
            const branchCard: RLMNode = {
              id,
              type: "branch_call",
              depth: e.depth,
              label: "Branch Call (rlm_query)",
              query: e.query,
              textSliceLen: e.text_slice_len,
              children: [],
              timestamp: Date.now(),
            };
            activeEngine.children.push(branchCard);
            activeBranchCalls[e.depth] = branchCard;
          } else if (e.type === "branch_end") {
            const branchCard = activeBranchCalls[e.depth];
            if (branchCard && branchCard.query === e.query) {
              branchCard.response = e.response;
            } else {
              const lastBranch = [...activeEngine.children]
                .reverse()
                .find((c) => c.type === "branch_call" && c.query === e.query);
              if (lastBranch) {
                lastBranch.response = e.response;
              }
            }
          }
        }
      }
    }
    return rootNodes;
  };

  const treeData = buildTree(events);

  // Recursive Tree Node Renderer Component
  const TreeNode = ({ node }: { node: RLMNode }) => {
    const isCollapsed = collapsedMap[node.id] || false;
    
    // Choose appropriate styling class based on node type
    let cardClass = styles.nodeCard;
    let nodeIconSymbol = "🌿";
    let badgeText = "Engine";

    if (node.type === "engine") {
      cardClass += ` ${styles.typeEngine}`;
      nodeIconSymbol = "⚙️";
      badgeText = `depth ${node.depth}`;
    } else if (node.type === "orchestrator") {
      cardClass += ` ${styles.typeOrchestrator}`;
      nodeIconSymbol = "🧠";
      badgeText = "Orchestrator";
    } else if (node.type === "executor") {
      cardClass += ` ${styles.typeExecutor}`;
      nodeIconSymbol = "💻";
      badgeText = "Executor";
    } else if (node.type === "leaf_call") {
      cardClass += ` ${styles.typeLeaf}`;
      nodeIconSymbol = "🍃";
      badgeText = "Leaf LLM";
    } else if (node.type === "branch_call") {
      cardClass += ` ${styles.typeBranch}`;
      nodeIconSymbol = "🌿";
      badgeText = "Branch RLM";
    }

    return (
      <div style={{ display: "flex", flexDirection: "column" }}>
        <div className={cardClass}>
          <div className={styles.nodeHeader} onClick={() => toggleCollapse(node.id)}>
            <div className={styles.nodeHeaderLeft}>
              <div className={styles.nodeIcon}>{nodeIconSymbol}</div>
              <div>
                <div className={styles.nodeTitle}>
                  {node.type === "engine" && `RLM Orchestrator Engine`}
                  {node.type === "orchestrator" && `Neural Thinking Process`}
                  {node.type === "executor" && `Sandbox Code Execution`}
                  {node.type === "leaf_call" && `Flat LLM Call (llm_query)`}
                  {node.type === "branch_call" && `Recursive Child RLM (rlm_query)`}
                </div>
                {node.query && (
                  <span style={{ fontSize: "12px", color: "var(--text-secondary)" }}>
                    "{node.query.length > 60 ? `${node.query.slice(0, 60)}...` : node.query}"
                  </span>
                )}
              </div>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <span className={styles.nodeBadge}>{badgeText}</span>
              {node.type !== "engine" && (
                <span className={`${styles.toggleIcon} ${!isCollapsed ? styles.toggleRotated : ""}`}>
                  ▼
                </span>
              )}
            </div>
          </div>

          {!isCollapsed && (
            <div className={styles.nodeBody}>
              {/* Engine Details */}
              {node.type === "engine" && (
                <>
                  <div>
                    <strong>Query:</strong> {node.query}
                  </div>
                  {node.contextLen !== undefined && (
                    <div>
                      <strong>Isolated Context Character Count:</strong> {node.contextLen.toLocaleString()}
                    </div>
                  )}
                  {node.finalAnswer && (
                    <div className={styles.nodeResponse} style={{ marginTop: "8px" }}>
                      <strong>Sub-Engine Result:</strong> {node.finalAnswer}
                    </div>
                  )}
                </>
              )}

              {/* Orchestrator Details */}
              {node.type === "orchestrator" && node.content && (() => {
                const { description, code } = parseOrchestratorContent(node.content);
                return (
                  <>
                    {description && (
                      <div style={{ whiteSpace: "pre-wrap" }}>
                        {description}
                      </div>
                    )}
                    {code && (
                      <div>
                        <div className={styles.codeBlockHeader}>
                          <span>Proposed Sandbox Code</span>
                          <span>Python</span>
                        </div>
                        <pre className={styles.codeBlock}>
                          <code>{code}</code>
                        </pre>
                      </div>
                    )}
                    {node.variablesSummary && (
                      <div className={styles.varsSummary}>
                        <div className={styles.varsSummaryTitle}>Active Variables Inventory</div>
                        <div style={{ whiteSpace: "pre-wrap" }}>{node.variablesSummary}</div>
                      </div>
                    )}
                    {node.finalAnswer && (
                      <div className={styles.nodeResponse}>
                        <strong>Converged to Final Answer:</strong> {node.finalAnswer}
                      </div>
                    )}
                  </>
                );
              })()}

              {/* Executor Details */}
              {node.type === "executor" && (
                <>
                  {node.code && (
                    <div>
                      <div className={styles.codeBlockHeader}>
                        <span>Executed Python Script</span>
                      </div>
                      <pre className={styles.codeBlock}>
                        <code>{node.code}</code>
                      </pre>
                    </div>
                  )}
                  {(node.stdout || node.stderr || node.exception) ? (
                    <div>
                      <div className={styles.codeBlockHeader}>
                        <span>Terminal Output</span>
                      </div>
                      <pre className={styles.terminalOutput}>
                        {node.stdout && <code>{node.stdout}</code>}
                        {node.stderr && <code className={styles.terminalStderr}>{node.stderr}</code>}
                        {node.exception && (
                          <div className={styles.terminalStderr}>
                            {node.exception}
                          </div>
                        )}
                      </pre>
                    </div>
                  ) : (
                    <div style={{ color: "var(--text-muted)", fontStyle: "italic" }}>
                      Code executed successfully with no stdout/stderr returned.
                    </div>
                  )}
                </>
              )}

              {/* Leaf Details */}
              {node.type === "leaf_call" && (
                <>
                  <div>
                    <strong>Sub-Query:</strong> {node.query}
                  </div>
                  {node.textSliceLen !== undefined && (
                    <div>
                      <strong>Text Slice Size:</strong> {node.textSliceLen.toLocaleString()} characters
                    </div>
                  )}
                  {node.response ? (
                    <div className={styles.nodeResponse}>
                      <strong>Leaf LLM Answer:</strong> {node.response}
                    </div>
                  ) : (
                    <div style={{ color: "var(--text-muted)", fontStyle: "italic" }}>
                      Awaiting response from worker...
                    </div>
                  )}
                </>
              )}

              {/* Branch Details */}
              {node.type === "branch_call" && (
                <>
                  <div>
                    <strong>Sub-Query:</strong> {node.query}
                  </div>
                  {node.textSliceLen !== undefined && (
                    <div>
                      <strong>Branch Slice Size:</strong> {node.textSliceLen.toLocaleString()} characters
                    </div>
                  )}
                  {node.response ? (
                    <div className={styles.nodeResponse}>
                      <strong>Recursive Answer:</strong> {node.response}
                    </div>
                  ) : (
                    <div style={{ color: "var(--text-muted)", fontStyle: "italic" }}>
                      Processing recursive subprocess...
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </div>

        {/* Nested Children */}
        {node.children && node.children.length > 0 && (
          <div className={styles.treeBranch}>
            {node.children.map((child) => (
              <TreeNode key={child.id} node={child} />
            ))}
          </div>
        )}
      </div>
    );
  };

  // Horizontal Mind-Map Node Renderer Component
  const GraphNodeView = ({ node, onClick }: { node: RLMNode; onClick: (n: RLMNode) => void }) => {
    const hasChildren = node.children && node.children.length > 0;
    
    let iconSymbol = "🌿";
    let badgeColor = "var(--accent-purple)";
    
    if (node.type === "engine") {
      iconSymbol = "⚙️";
      badgeColor = "var(--accent-purple)";
    } else if (node.type === "orchestrator") {
      iconSymbol = "🧠";
      badgeColor = "#c084fc";
    } else if (node.type === "executor") {
      iconSymbol = "💻";
      badgeColor = "var(--accent-green)";
    } else if (node.type === "leaf_call") {
      iconSymbol = "🍃";
      badgeColor = "var(--accent-blue)";
    } else if (node.type === "branch_call") {
      iconSymbol = "🌿";
      badgeColor = "#fb7185";
    }

    // Short summary text inside node graph box
    let boxText = "Click for details";
    if (node.type === "orchestrator" && node.content) {
      const { description } = parseOrchestratorContent(node.content);
      boxText = description;
    } else if (node.type === "executor" && node.code) {
      boxText = node.code;
    } else if (node.type === "leaf_call" && node.query) {
      boxText = node.query;
    } else if (node.type === "branch_call" && node.query) {
      boxText = node.query;
    } else if (node.type === "engine" && node.query) {
      boxText = node.query;
    }

    return (
      <div className={styles.graphNode}>
        <div 
          className={`${styles.nodeBox} ${hasChildren ? styles.nodeBoxHasChildren : ""}`}
          onClick={() => onClick(node)}
        >
          <div className={styles.nodeBoxTitle}>
            <span>{iconSymbol}</span>
            <span>
              {node.type === "engine" && `Engine (Depth ${node.depth})`}
              {node.type === "orchestrator" && `Orchestrator`}
              {node.type === "executor" && `Executor`}
              {node.type === "leaf_call" && `Leaf Call`}
              {node.type === "branch_call" && `Branch Call`}
            </span>
          </div>
          <div className={styles.nodeBoxText}>
            {boxText}
          </div>
          <span 
            className={styles.nodeBoxBadge} 
            style={{ color: badgeColor, borderColor: `${badgeColor}33` }}
          >
            {node.type}
          </span>
        </div>

        {hasChildren && (
          <div className={styles.graphChildren}>
            {node.children.map((child) => (
              <div key={child.id} className={styles.graphChildWrapper}>
                <GraphNodeView node={child} onClick={onClick} />
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className={styles.container}>
      {/* Header Panel */}
      <header className={styles.header}>
        <div className={styles.titleArea}>
          <h1>Recursive Language Model (RLM) Visualizer</h1>
          <p>Isolating text in stateful sandboxes with recursive call structures</p>
        </div>
        <div className={styles.statusIndicator}>
          <div
            className={`${styles.statusDot} ${
              status === "running"
                ? styles.statusRunning
                : status === "success"
                ? styles.statusSuccess
                : status === "error"
                ? styles.statusError
                : ""
            }`}
          />
          <span style={{ textTransform: "capitalize", fontWeight: "600" }}>
            {status === "idle" ? "ready" : status}
          </span>
        </div>
      </header>

      {/* Main Content Layout */}
      <div className={styles.layout}>
        {/* Left column: Inputs Panel */}
        <aside className={`${styles.sidebar} glowing-panel`}>
          <div className={styles.sectionTitle}>Pipeline Settings</div>
          
          <form onSubmit={handleRun} style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
            {/* Provider Select */}
            <div className={styles.formGroup}>
              <label>LLM Provider</label>
              <select
                className={styles.select}
                value={provider}
                onChange={(e) => setProvider(e.target.value)}
                disabled={isRunning}
              >
                <option value="mock">Simulation Mock Model</option>
                <option value="openai">OpenAI (requires key)</option>
                <option value="anthropic">Anthropic (requires key)</option>
                <option value="google">Google GenAI (requires key)</option>
              </select>
            </div>

            {/* Model Name Input */}
            <div className={styles.formGroup}>
              <label>Model Name</label>
              <input
                type="text"
                className={styles.input}
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                disabled={isRunning}
                placeholder="e.g. gpt-4o-mini"
              />
            </div>

            {/* Max Depth and Steps */}
            <div className={styles.rowInputs}>
              <div className={styles.formGroup}>
                <label>Max Depth</label>
                <input
                  type="number"
                  className={styles.input}
                  value={maxDepth}
                  onChange={(e) => setMaxDepth(parseInt(e.target.value) || 1)}
                  disabled={isRunning}
                  min="1"
                  max="5"
                />
              </div>
              <div className={styles.formGroup}>
                <label>Max Steps</label>
                <input
                  type="number"
                  className={styles.input}
                  value={maxSteps}
                  onChange={(e) => setMaxSteps(parseInt(e.target.value) || 1)}
                  disabled={isRunning}
                  min="1"
                  max="30"
                />
              </div>
            </div>

            {/* Context Source Toggle */}
            <div className={styles.formGroup}>
              <label>Context Input Type</label>
              <div style={{ display: "flex", gap: "12px", marginTop: "4px" }}>
                <label style={{ display: "flex", alignItems: "center", gap: "6px", cursor: "pointer" }}>
                  <input
                    type="radio"
                    checked={contextSource === "text"}
                    onChange={() => setContextSource("text")}
                    disabled={isRunning}
                  />
                  Raw Text
                </label>
                <label style={{ display: "flex", alignItems: "center", gap: "6px", cursor: "pointer" }}>
                  <input
                    type="radio"
                    checked={contextSource === "file"}
                    onChange={() => setContextSource("file")}
                    disabled={isRunning}
                  />
                  PDF / Text File
                </label>
              </div>
            </div>

            {/* Context Value Input */}
            {contextSource === "text" ? (
              <div className={styles.formGroup}>
                <label>Context Document Content</label>
                <textarea
                  className={styles.textarea}
                  value={contextText}
                  onChange={(e) => setContextText(e.target.value)}
                  disabled={isRunning}
                  rows={6}
                  placeholder="Paste long text context here..."
                />
              </div>
            ) : (
              <div className={styles.formGroup}>
                <label>Upload Document</label>
                <div className={styles.fileUploadArea}>
                  <div className={styles.uploadIcon}>📁</div>
                  <div className={styles.uploadText}>
                    {file ? file.name : "Drag & drop file or click to browse"}
                  </div>
                  {file && (
                    <div className={styles.fileSelected}>
                      {(file.size / 1024).toFixed(1)} KB
                    </div>
                  )}
                  <input
                    type="file"
                    ref={fileInputRef}
                    className={styles.fileInput}
                    accept=".pdf,.txt"
                    onChange={(e) => {
                      if (e.target.files && e.target.files[0]) {
                        setFile(e.target.files[0]);
                      }
                    }}
                    disabled={isRunning}
                  />
                </div>
              </div>
            )}

            {/* Query Input */}
            <div className={styles.formGroup}>
              <label>User Query</label>
              <textarea
                className={styles.textarea}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                disabled={isRunning}
                rows={3}
                placeholder="Ask something about the context..."
              />
            </div>

            {/* Run / Stop Buttons */}
            {isRunning ? (
              <button type="button" className={styles.btnStop} onClick={handleStop}>
                ⏹ Stop Execution
              </button>
            ) : (
              <button
                type="submit"
                className={styles.btnRun}
                disabled={
                  query.trim() === "" ||
                  (contextSource === "text" ? contextText.trim() === "" : !file)
                }
              >
                ⚡ Execute Query
              </button>
            )}
          </form>

          {/* Statistics Panel */}
          <div className={styles.statsCard}>
            <div className={styles.statItem}>
              <span className={styles.statLabel}>Context Size</span>
              <span className={styles.statVal}>
                {contextSource === "text"
                  ? `${characterCount.toLocaleString()} chars`
                  : file
                  ? `${(characterCount / 1024).toFixed(1)} KB`
                  : "0"}
              </span>
            </div>
            <div className={styles.statItem}>
              <span className={styles.statLabel}>Elapsed Time</span>
              <span className={styles.statVal}>{elapsedTime}s</span>
            </div>
            <div className={styles.statItem} style={{ marginTop: "8px" }}>
              <span className={styles.statLabel}>Total Steps</span>
              <span className={styles.statVal}>{totalSteps}</span>
            </div>
            <div className={styles.statItem} style={{ marginTop: "8px" }}>
              <span className={styles.statLabel}>Active Depth</span>
              <span className={styles.statVal}>
                {events.length > 0 ? Math.max(...events.map((ev) => ev.depth || 0)) : 0}
              </span>
            </div>
          </div>
        </aside>

        {/* Right column: Execution Logs & Visualizer Tree */}
        <main className={`${styles.visualizerPanel} glowing-panel`}>
          <div className={styles.panelHeader}>
            <span className={styles.panelTitle}>Execution Visualizer</span>
            
            {/* View Mode Tabs */}
            {events.length > 0 && (
              <div className={styles.tabContainer}>
                <button
                  type="button"
                  className={`${styles.tabButton} ${viewMode === "tree" ? styles.tabButtonActive : ""}`}
                  onClick={() => setViewMode("tree")}
                >
                  Timeline Tree
                </button>
                <button
                  type="button"
                  className={`${styles.tabButton} ${viewMode === "graph" ? styles.tabButtonActive : ""}`}
                  onClick={() => setViewMode("graph")}
                >
                  Node Graph
                </button>
              </div>
            )}

            {events.length > 0 && (
              <button className={styles.clearBtn} onClick={handleClear} disabled={isRunning}>
                Clear Output
              </button>
            )}
          </div>

          {errorMsg && (
            <div
              style={{
                background: "rgba(248, 113, 113, 0.08)",
                border: "1px solid rgba(248, 113, 113, 0.3)",
                color: "var(--accent-red)",
                padding: "12px",
                borderRadius: "8px",
                fontSize: "14px",
              }}
            >
              <strong>Error:</strong> {errorMsg}
            </div>
          )}

          {/* Visualizer output tree */}
          <div style={{ flex: 1, minWidth: 0 }}>
            {events.length === 0 ? (
              <div className={styles.emptyState}>
                <div className={styles.emptyStateIcon}>🧬</div>
                <div>Awaiting query execution...</div>
                <div style={{ fontSize: "12px", color: "var(--text-muted)", maxWidth: "300px", textAlign: "center" }}>
                  Run the simulation mock model or configure API keys to run queries over your documents.
                </div>
              </div>
            ) : viewMode === "tree" ? (
              <div className={styles.treeContainer}>
                {treeData.map((node) => (
                  <TreeNode key={node.id} node={node} />
                ))}
              </div>
            ) : (
              <div className={styles.graphContainer}>
                <div className={styles.graphWrapper}>
                  {treeData.map((node) => (
                    <GraphNodeView key={node.id} node={node} onClick={setSelectedNode} />
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Final Answer Panel */}
          {finalAnswer && (
            <div className={styles.finalAnswerPanel}>
              <div className={styles.finalAnswerHeader}>
                <span>✨ Final Result</span>
              </div>
              <div className={styles.finalAnswerBody}>{finalAnswer}</div>
            </div>
          )}
        </main>
      </div>

      {/* Modal detail overlay */}
      {selectedNode && (
        <div className={styles.modalOverlay} onClick={() => setSelectedNode(null)}>
          <div className={styles.modalContent} onClick={(e) => e.stopPropagation()}>
            <button className={styles.modalCloseBtn} onClick={() => setSelectedNode(null)}>
              ✕
            </button>
            <div style={{ padding: "8px" }}>
              <TreeNode node={{ ...selectedNode, children: [] }} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

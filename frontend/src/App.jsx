import React, { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { 
  Search, Compass, UserCheck, HelpCircle, ArrowRight, Check, 
  AlertTriangle, AlertCircle, RefreshCw, Plus, CheckCircle, 
  ExternalLink, Calendar, Mail, FileText, Send, GitBranch, 
  MapPin, Database, Award, Info, Trash2, CheckCircle2, ChevronRight, X
} from 'lucide-react';

export default function App() {
  // Navigation Tabs: 'Dashboard', 'Relationships', 'Pipeline', 'Universities', 'Workspace'
  const [activeTab, setActiveTab] = useState('Dashboard');
  const [workspaceSubTab, setWorkspaceSubTab] = useState('Investigations');
  
  // Global KPIs
  const [kpis, setKpis] = useState({
    active_investigations: 0,
    open_blockers: 0,
    programs_indexed: 0,
    professors_contacted: 0,
    replies_received: 0,
    active_conversations: 0,
    internship_candidates: 0
  });

  // Data Stores
  const [professors, setProfessors] = useState([]);
  const [selectedProf, setSelectedProf] = useState(null);
  const [dashboardData, setDashboardData] = useState({
    hero_actions: [],
    follow_ups: [],
    lost_opportunities: [],
    top_opportunities: [],
    dormant_relationships: []
  });
  const [universities, setUniversities] = useState([]);
  const [selectedUni, setSelectedUni] = useState(null);
  const [investigations, setInvestigations] = useState([]);
  const [selfModel, setSelfModel] = useState({ strong_claims: [], open_questions: [], methods_used: [], topics: [] });
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [selectedNode, setSelectedNode] = useState(null);

  // Active interaction selection for replies
  const [activeInteraction, setActiveInteraction] = useState(null);
  
  // Modals & UI States
  const [replyText, setReplyText] = useState('');
  const [isExtracting, setIsExtracting] = useState(false);
  const [extractionModalOpen, setExtractionModalOpen] = useState(false);
  const [extractedIntel, setExtractedIntel] = useState({
    suggested_paper: '',
    paper_confidence: 0,
    suggested_experiment: '',
    experiment_confidence: 0
  });
  const [paperChecked, setPaperChecked] = useState(true);
  const [experimentChecked, setExperimentChecked] = useState(true);
  const [scoreModalOpen, setScoreModalOpen] = useState(false);
  const [scoreExplProf, setScoreExplProf] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  
  // Live Analysis Console States
  const [showLogsModal, setShowLogsModal] = useState(false);
  const [taskLogs, setTaskLogs] = useState([]);
  const [activeTaskId, setActiveTaskId] = useState(null);
  const [taskStatus, setTaskStatus] = useState('idle'); // idle, running, completed, failed, stopped
  const [isLogsMinimized, setIsLogsMinimized] = useState(false); // minimize log overlay to bottom tray
  const [liteScan, setLiteScan] = useState(false); // lite scan mode: skip PDFs
  const [selectedQuestionIdx, setSelectedQuestionIdx] = useState(0); // index into questions_worth_asking
  
  const [profProfile, setProfProfile] = useState(null);
  const [dossierTab, setDossierTab] = useState('synthesis');
  const [expandedPaperIdx, setExpandedPaperIdx] = useState(null);

  useEffect(() => {
    if (!selectedProf) {
      setProfProfile(null);
      setExpandedPaperIdx(null);
      return;
    }
    setExpandedPaperIdx(null);
    setDossierTab('synthesis');
    
    const fetchProfile = async () => {
      try {
        const cleanName = selectedProf.professor_name
          .toLowerCase()
          .replace(/[^a-z0-9_ ]/g, '')
          .trim()
          .replace(/\s+/g, '_');
        const res = await fetch(`/reports/${cleanName}/${cleanName}_profile.json`);
        if (res.ok) {
          const data = await res.json();
          setProfProfile(data);
        } else {
          setProfProfile(null);
        }
      } catch (e) {
        console.error("Error fetching full profile JSON", e);
        setProfProfile(null);
      }
    };
    fetchProfile();
  }, [selectedProf]);

  // Refs
  const graphRef = useRef(null);
  const consoleRef = useRef(null);
  const eventSourceRef = useRef(null);

  // Load KPIs
  const loadKpis = async () => {
    try {
      const res = await fetch('/api/crm/kpis');
      const data = await res.json();
      setKpis(data);
    } catch (e) {
      console.error("Error loading KPIs", e);
    }
  };

  // Load Professors
  const loadProfessors = async (selectId = null) => {
    try {
      const res = await fetch('/api/crm/professors');
      const data = await res.json();
      setProfessors(data);
      if (data.length > 0) {
        if (selectId) {
          const matched = data.find(p => (p.openalex_id || p.professor_name) === selectId);
          if (matched) setSelectedProf(matched);
        } else if (!selectedProf) {
          setSelectedProf(data[0]);
        } else {
          // Keep current selection updated
          const updated = data.find(p => (p.openalex_id || p.professor_name) === (selectedProf.openalex_id || selectedProf.professor_name));
          if (updated) setSelectedProf(updated);
        }
      }
    } catch (e) {
      console.error("Error loading professors", e);
    }
  };

  // Load Dashboard Summary
  const loadDashboard = async () => {
    try {
      const res = await fetch('/api/crm/dashboard_summary');
      const data = await res.json();
      setDashboardData(data);
    } catch (e) {
      console.error("Error loading dashboard", e);
    }
  };

  // Load Universities List
  const loadUniversities = async (selectName = null) => {
    try {
      const res = await fetch('/api/crm/universities');
      const data = await res.json();
      setUniversities(data);
      if (data.length > 0) {
        if (selectName) {
          const matched = data.find(u => u.name === selectName);
          if (matched) setSelectedUni(matched);
        } else if (!selectedUni) {
          setSelectedUni(data[0]);
        } else {
          const matched = data.find(u => u.name === selectedUni.name);
          if (matched) setSelectedUni(matched);
        }
      }
    } catch (e) {
      console.error("Error loading universities", e);
    }
  };

  // Load Workspace supporting data
  const loadWorkspaceData = async () => {
    try {
      const resInvs = await fetch('/api/crm/investigations');
      const dataInvs = await resInvs.json();
      setInvestigations(dataInvs);

      const resSelf = await fetch('/api/crm/self_model');
      const dataSelf = await resSelf.json();
      setSelfModel(dataSelf);

      const resGraph = await fetch('/api/crm/unified_graph');
      const dataGraph = await resGraph.json();
      setGraphData(dataGraph);
    } catch (e) {
      console.error("Error loading workspace data", e);
    }
  };

  // Run on mount
  useEffect(() => {
    loadKpis();
    loadProfessors();
    loadDashboard();
    loadUniversities();
    loadWorkspaceData();
  }, []);

  // Scroll logs to bottom automatically
  useEffect(() => {
    if (consoleRef.current) {
      consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
    }
  }, [taskLogs, showLogsModal]);

  // Clean up EventSource connection on unmount
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  const checkTaskStatusDirectly = async (taskId) => {
    try {
      const res = await fetch('/api/active_tasks');
      const active = await res.json();
      const isRunning = active.some(t => t.task_id === taskId);
      if (!isRunning) {
        setTaskStatus(prev => prev === 'running' ? 'completed' : prev);
        if (eventSourceRef.current) {
          eventSourceRef.current.close();
          eventSourceRef.current = null;
        }
        // Refresh active views
        if (selectedProf) {
          loadProfessors(selectedProf.openalex_id || selectedProf.professor_name);
        } else {
          loadProfessors();
        }
        loadDashboard();
        loadKpis();
        loadWorkspaceData();
      }
    } catch (e) {
      console.error("Error checking task status directly", e);
    }
  };

  const startLogsStream = (taskId) => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }
    
    const source = new EventSource(`/api/stream/${taskId}`);
    eventSourceRef.current = source;
    
    source.addEventListener('log', (event) => {
      try {
        const data = JSON.parse(event.data);
        setTaskLogs(prev => [...prev, data.line]);
      } catch (e) {
        console.error("Error parsing stream log event", e);
      }
    });
    
    source.addEventListener('status', (event) => {
      try {
        const data = JSON.parse(event.data);
        setTaskStatus(data.status);
        source.close();
        eventSourceRef.current = null;
        
        // Refresh professors and related data
        if (selectedProf) {
          loadProfessors(selectedProf.openalex_id || selectedProf.professor_name);
        } else {
          loadProfessors();
        }
        loadDashboard();
        loadKpis();
        loadWorkspaceData();
        loadUniversities();
      } catch (e) {
        console.error("Error parsing stream status event", e);
      }
    });
    
    source.onerror = (err) => {
      console.error("EventSource error:", err);
      checkTaskStatusDirectly(taskId);
    };
  };

  const triggerDeepAnalysis = async (profName, useLiteScan = false) => {
    const scanLabel = useLiteScan ? 'lite scan' : 'deep analysis';
    setTaskLogs([`[System] Initializing ${scanLabel} for ${profName}...`]);
    setTaskStatus('running');
    setShowLogsModal(true);
    setIsLogsMinimized(false);
    setActiveTaskId(null);
    
    try {
      const res = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: 'batch',
          batch_input: profName,
          deep_run: !useLiteScan,
          lite_scan: useLiteScan,
          limit_papers: useLiteScan ? 3 : 5
        })
      });
      const data = await res.json();
      if (data.task_id) {
        setActiveTaskId(data.task_id);
        startLogsStream(data.task_id);
      } else {
        setTaskLogs(prev => [...prev, `[Error] Failed to trigger analysis: ${data.error || 'Unknown error'}`]);
        setTaskStatus('failed');
      }
    } catch (e) {
      console.error(e);
      setTaskLogs(prev => [...prev, `[Error] Failed to connect to server: ${e.message}`]);
      setTaskStatus('failed');
    }
  };

  const triggerUniversityDiscovery = async (uniName, useLiteScan = false) => {
    const scanLabel = useLiteScan ? 'lite scan' : 'deep analysis';
    setTaskLogs([`[System] Initializing university ${scanLabel} for ${uniName}...`]);
    setTaskStatus('running');
    setShowLogsModal(true);
    setIsLogsMinimized(false);
    setActiveTaskId(null);
    
    try {
      const res = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: 'university',
          university: uniName,
          lite_scan: useLiteScan,
          discovery_mode: useLiteScan ? 'internship' : 'research',
          limit_professors: 20
        })
      });
      const data = await res.json();
      if (data.task_id) {
        setActiveTaskId(data.task_id);
        startLogsStream(data.task_id);
      } else {
        setTaskLogs(prev => [...prev, `[Error] Failed to trigger discovery: ${data.error || 'Unknown error'}`]);
        setTaskStatus('failed');
      }
    } catch (e) {
      console.error(e);
      setTaskLogs(prev => [...prev, `[Error] Failed to connect to server: ${e.message}`]);
      setTaskStatus('failed');
    }
  };

  const stopAnalysis = async () => {
    if (!activeTaskId) return;
    setTaskLogs(prev => [...prev, `[System] Sending termination request...`]);
    try {
      const res = await fetch(`/api/stop/${activeTaskId}`, {
        method: 'POST'
      });
      const data = await res.json();
      setTaskStatus('stopped');
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      if (selectedProf) {
        loadProfessors(selectedProf.openalex_id || selectedProf.professor_name);
      } else {
        loadProfessors();
      }
      loadDashboard();
      loadKpis();
      loadWorkspaceData();
    } catch (e) {
      console.error("Error stopping task", e);
      setTaskLogs(prev => [...prev, `[Error] Failed to stop task: ${e.message}`]);
    }
  };

  // Update unified graph D3 canvas
  useEffect(() => {
    if (activeTab === 'Workspace' && workspaceSubTab === 'Graph' && graphRef.current && graphData.nodes.length > 0) {
      drawGraph(graphData);
    }
  }, [activeTab, workspaceSubTab, graphData]);

  // Handle Search Input
  const handleSearch = async (val) => {
    setSearchQuery(val);
    if (!val.trim()) {
      setSearchResults([]);
      return;
    }
    try {
      const res = await fetch(`/api/crm/search?q=${encodeURIComponent(val)}`);
      const data = await res.json();
      setSearchResults(data);
    } catch (e) {
      console.error("Error querying search", e);
    }
  };

  // Change professor status manually
  const handleStatusChange = async (profId, newStatus) => {
    try {
      const res = await fetch(`/api/crm/professors/${encodeURIComponent(profId)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: newStatus })
      });
      if (res.ok) {
        await loadProfessors(profId);
        loadDashboard();
        loadKpis();
        loadUniversities();
      }
    } catch (e) {
      console.error("Error changing status", e);
    }
  };

  // Log reply intelligence extraction
  const triggerReplyExtraction = async (interaction) => {
    if (!replyText.trim()) return;
    setIsExtracting(true);
    try {
      const res = await fetch('/api/crm/extract_intelligence', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email_body: replyText })
      });
      const data = await res.json();
      setExtractedIntel(data);
      setPaperChecked(!!data.suggested_paper);
      setExperimentChecked(!!data.suggested_experiment);
      setActiveInteraction(interaction);
      setExtractionModalOpen(true);
    } catch (e) {
      console.error("Error extracting intelligence", e);
    } finally {
      setIsExtracting(false);
    }
  };

  // Confirm tasks and log reply
  const handleConfirmTasks = async () => {
    if (!selectedProf || !activeInteraction) return;
    const profId = selectedProf.openalex_id || selectedProf.professor_name;
    
    try {
      // 1. Confirm and create tasks (Outcomes)
      await fetch('/api/crm/confirm_tasks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prof_id: profId,
          investigation_id: activeInteraction.investigation_id,
          blocker_id: activeInteraction.blocker_id,
          suggested_paper: extractedIntel.suggested_paper,
          suggested_paper_confirmed: paperChecked,
          suggested_experiment: extractedIntel.suggested_experiment,
          suggested_experiment_confirmed: experimentChecked
        })
      });

      // 2. Mark interaction as replied
      await fetch(`/api/crm/interactions/${activeInteraction.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'reply',
          response: replyText,
          evidence_gained: extractedIntel.suggested_paper || extractedIntel.suggested_experiment || 'Replied and processed suggested tasks.'
        })
      });

      // Reset
      setReplyText('');
      setExtractionModalOpen(false);
      loadProfessors(profId);
      loadDashboard();
      loadKpis();
      loadUniversities();
    } catch (e) {
      console.error("Error confirming tasks", e);
    }
  };

  // Focus a professor from action
  const focusProfessor = (profId) => {
    const prof = professors.find(p => (p.openalex_id || p.professor_name) === profId);
    if (prof) {
      setSelectedProf(prof);
      setActiveTab('Relationships');
    }
  };

  // Drag and Drop Pipeline column move
  const movePIStatus = async (profId, nextStatus) => {
    await handleStatusChange(profId, nextStatus);
  };

  // D3 force graph layout renderer
  const drawGraph = (data) => {
    if (!graphRef.current) return;
    const svg = d3.select(graphRef.current);
    svg.selectAll("*").remove();
    
    const width = graphRef.current.clientWidth || 800;
    const height = graphRef.current.clientHeight || 550;
    
    const container = svg.append("g");
    
    svg.call(d3.zoom().on("zoom", (event) => {
      container.attr("transform", event.transform);
    }));

    svg.append("defs").append("marker")
      .attr("id", "arrow")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 18)
      .attr("refY", 0)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", "rgba(255,255,255,0.2)");

    const nodes = data.nodes;
    const links = data.links;
    
    const colorMap = {
      investigation: "var(--accent-emerald)",
      blocker: "var(--accent-rose)",
      program: "var(--accent-violet)",
      professor: "#ffffff",
      claim: "var(--accent-cyan)",
      outcome: "var(--accent-amber)",
      paper: "var(--text-muted)"
    };

    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => d.id).distance(110))
      .force("charge", d3.forceManyBody().strength(-220))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(35));

    const link = container.append("g")
      .selectAll("line")
      .data(links)
      .enter().append("line")
      .attr("stroke", "rgba(255,255,255,0.15)")
      .attr("stroke-width", 1.5)
      .attr("marker-end", "url(#arrow)");

    const node = container.append("g")
      .selectAll("g")
      .data(nodes)
      .enter().append("g")
      .call(d3.drag()
        .on("start", (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on("drag", (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on("end", (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        }))
      .on("click", (event, d) => {
        event.stopPropagation();
        setSelectedNode(d);
      });

    node.append("circle")
      .attr("r", d => d.type === 'investigation' ? 14 : 10)
      .attr("fill", d => colorMap[d.type] || "#ffffff")
      .attr("stroke", "rgba(255,255,255,0.2)")
      .attr("stroke-width", 2)
      .style("filter", d => `drop-shadow(0px 0px 6px ${colorMap[d.type] || '#fff'})`);

    node.append("text")
      .text(d => d.label.length > 20 ? d.label.substring(0, 18) + "..." : d.label)
      .attr("x", 14)
      .attr("y", 4)
      .attr("fill", "var(--text-secondary)")
      .style("font-size", "0.75rem")
      .style("pointer-events", "none");

    simulation.on("tick", () => {
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      node
        .attr("transform", d => `translate(${d.x}, ${d.y})`);
    });
  };

  return (
    <>
      {/* App Header */}
      <header className="app-header">
        <div className="brand">
          <span className="brand-logo">Research Relationship OS</span>
          <span className="brand-badge">Engine</span>
        </div>

        {/* Global Search Bar */}
        <div style={{ position: 'relative', width: '320px' }}>
          <div style={{ display: 'flex', alignItems: 'center', background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border-color)', borderRadius: '6px', padding: '0.4rem 0.75rem' }}>
            <Search size={16} color="var(--text-muted)" style={{ marginRight: '0.5rem' }} />
            <input 
              type="text" 
              placeholder="Search PIs, claims, programs..." 
              value={searchQuery}
              onChange={(e) => handleSearch(e.target.value)}
              style={{ background: 'transparent', border: 'none', color: '#fff', outline: 'none', fontSize: '0.82rem', width: '100%' }}
            />
            {searchQuery && <X size={14} color="var(--text-muted)" style={{ cursor: 'pointer' }} onClick={() => handleSearch('')} />}
          </div>
          {/* Search Dropdown */}
          {searchResults.length > 0 && (
            <div style={{ position: 'absolute', top: '100%', left: 0, right: 0, background: 'var(--bg-surface)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: '8px', zIndex: 120, marginTop: '0.5rem', maxHeight: '300px', overflowY: 'auto', boxShadow: '0 8px 24px rgba(0,0,0,0.5)' }}>
              {searchResults.map((item, idx) => (
                <div 
                  key={idx} 
                  style={{ padding: '0.75rem 1rem', borderBottom: '1px solid rgba(255,255,255,0.05)', cursor: 'pointer' }}
                  onClick={() => {
                    setSearchQuery('');
                    setSearchResults([]);
                    if (item.type === 'professor') {
                      focusProfessor(item.id);
                    } else if (item.type === 'investigation') {
                      setActiveTab('Workspace');
                      setWorkspaceSubTab('Investigations');
                    } else if (item.type === 'program') {
                      setActiveTab('Universities');
                    }
                  }}
                >
                  <div style={{ fontSize: '0.85rem', color: '#fff', fontWeight: 600 }}>{item.label}</div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.15rem' }}>{item.sublabel}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Navigation Tabs */}
        <nav className="nav-tabs">
          {['Dashboard', 'Relationships', 'Pipeline', 'Universities', 'Workspace'].map(tab => (
            <button 
              key={tab} 
              className={`nav-tab-btn ${activeTab === tab ? 'active' : ''}`}
              onClick={() => setActiveTab(tab)}
            >
              {tab === 'Dashboard' && <Compass size={16} />}
              {tab === 'Relationships' && <UserCheck size={16} />}
              {tab === 'Pipeline' && <GitBranch size={16} />}
              {tab === 'Universities' && <MapPin size={16} />}
              {tab === 'Workspace' && <Database size={16} />}
              {tab === 'Workspace' ? 'Research Workspace' : tab}
            </button>
          ))}
        </nav>
      </header>

      {/* Global KPIs Bar */}
      <div className="kpi-row">
        <div className="kpi-card">
          <div className="kpi-val" style={{ color: 'var(--accent-emerald)' }}>{kpis.active_investigations}</div>
          <div className="kpi-lbl">Active Invs</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-val" style={{ color: 'var(--accent-rose)' }}>{kpis.open_blockers}</div>
          <div className="kpi-lbl">Open Blockers</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-val" style={{ color: 'var(--accent-violet)' }}>{kpis.programs_indexed}</div>
          <div className="kpi-lbl">Programs Indexed</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-val" style={{ color: '#fff' }}>{kpis.professors_contacted}</div>
          <div className="kpi-lbl">PIs Contacted</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-val" style={{ color: 'var(--accent-cyan)' }}>{kpis.replies_received}</div>
          <div className="kpi-lbl">Replies Received</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-val" style={{ color: 'var(--accent-amber)' }}>{kpis.internship_candidates}</div>
          <div className="kpi-lbl">Candidates</div>
        </div>
      </div>

      {/* Main Workspace content */}
      <main className="app-main">
        {/* DASHBOARD TAB */}
        {activeTab === 'Dashboard' && (
          <div className="flex-row">
            {/* Left: What Should I Do Today? Hero panel (70% width) */}
            <div className="glass-panel" style={{ flex: '2', minWidth: '600px' }}>
              <div className="panel-title" style={{ fontSize: '1.2rem', color: 'var(--accent-cyan)', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.75rem' }}>
                <CheckCircle2 size={18} color="var(--accent-cyan)" /> What Should I Do Today?
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: '1rem' }}>
                {dashboardData.hero_actions.map((act, idx) => (
                  <div key={idx} style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-color)', borderRadius: '10px', padding: '1.25rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', transition: 'border-color 0.2s', hover: { borderColor: 'var(--border-hover)' } }}>
                    <div style={{ flex: 1, paddingRight: '1rem' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.35rem' }}>
                        <span style={{ fontSize: '1rem', fontWeight: 700, color: '#fff' }}>{idx + 1}. {act.title}</span>
                        <span className="badge" style={{ background: act.priority === 'High' ? 'rgba(244,63,94,0.1)' : 'rgba(245,158,11,0.1)', color: act.priority === 'High' ? 'var(--accent-rose)' : 'var(--accent-amber)', border: '1px solid transparent' }}>{act.priority} Priority</span>
                      </div>
                      <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>{act.reason}</div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--accent-cyan)', marginTop: '0.5rem', fontWeight: 600 }}>Expected Impact: {act.expected_impact}</div>
                    </div>
                    {act.prof_id && (
                      <button className="btn-primary" onClick={() => focusProfessor(act.prof_id)}>
                        Action <ArrowRight size={14} />
                      </button>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Right: Side columns (30% width) */}
            <div style={{ flex: '1', minWidth: '320px', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              {/* Opportunities Being Lost */}
              <div className="glass-panel" style={{ borderLeft: '3px solid var(--accent-rose)' }}>
                <div className="panel-title" style={{ color: 'var(--accent-rose)', fontSize: '0.9rem' }}>
                  <AlertTriangle size={16} color="var(--accent-rose)" /> Opportunities Being Lost
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem' }}>
                  {dashboardData.lost_opportunities.length === 0 ? (
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>No high-risk dormant leads detected. Keep it up!</div>
                  ) : (
                    dashboardData.lost_opportunities.map((opp, idx) => (
                      <div key={idx} style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid var(--border-color)', borderRadius: '6px', padding: '0.65rem 0.85rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div>
                          <div style={{ fontSize: '0.82rem', fontWeight: 600, color: '#fff' }}>{opp.professor_name}</div>
                          <div style={{ fontSize: '0.72rem', color: 'var(--accent-rose)' }}>{opp.reason}</div>
                        </div>
                        <button className="btn-secondary" style={{ padding: '0.3rem 0.6rem', fontSize: '0.7rem' }} onClick={() => focusProfessor(opp.prof_id)}>Rescue</button>
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Follow-ups Due */}
              <div className="glass-panel">
                <div className="panel-title" style={{ fontSize: '0.9rem' }}>
                  <Calendar size={16} /> Follow-ups Due
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem' }}>
                  {dashboardData.follow_ups.length === 0 ? (
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>No followups due.</div>
                  ) : (
                    dashboardData.follow_ups.map((f, idx) => (
                      <div key={idx} style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid var(--border-color)', borderRadius: '6px', padding: '0.65rem' }} onClick={() => focusProfessor(f.prof_id)}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', fontWeight: 600, color: '#fff' }}>
                          <span>{f.professor_name}</span>
                          <span style={{ color: 'var(--accent-violet)' }}>{f.follow_up_date}</span>
                        </div>
                        <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>{f.reason}</div>
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Dormant */}
              <div className="glass-panel">
                <div className="panel-title" style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>
                  <Info size={16} color="var(--text-muted)" /> Dormant Relationships
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem' }}>
                  {dashboardData.dormant_relationships.length === 0 ? (
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>No dormant relationships.</div>
                  ) : (
                    dashboardData.dormant_relationships.map((d, idx) => (
                      <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.8rem', padding: '0.25rem 0' }}>
                        <span style={{ color: 'var(--text-secondary)' }}>{d.professor_name} ({d.institution.substring(0, 15)}...)</span>
                        <span style={{ color: 'var(--accent-rose)' }}>Readiness: {d.readiness}</span>
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Top Opportunities */}
              <div className="glass-panel">
                <div className="panel-title" style={{ fontSize: '0.9rem', color: 'var(--accent-emerald)' }}>
                  <Award size={16} color="var(--accent-emerald)" /> Top Leads
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem' }}>
                  {dashboardData.top_opportunities.map((t, idx) => (
                    <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.8rem', padding: '0.25rem 0', cursor: 'pointer' }} onClick={() => focusProfessor(t.prof_id)}>
                      <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>{t.professor_name}</span>
                      <span style={{ color: 'var(--accent-emerald)' }}>Readiness: {t.readiness}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* RELATIONSHIPS TAB */}
        {activeTab === 'Relationships' && (
          <div className="sidebar-layout">
            {/* Left: Sidebar PIs list */}
            <div className="glass-panel sidebar-panel">
              <div className="panel-title">Profiled Relationships</div>
              <div className="sidebar-list">
                {professors.map((p, idx) => {
                  const id = p.openalex_id || p.professor_name;
                  const isActive = selectedProf && (selectedProf.openalex_id || selectedProf.professor_name) === id;
                  return (
                    <button 
                      key={idx}
                      className={`list-item-btn ${isActive ? 'active' : ''}`}
                      onClick={() => setSelectedProf(p)}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
                        <strong style={{ color: '#fff', fontSize: '0.88rem' }}>{p.professor_name}</strong>
                        <span style={{ color: 'var(--accent-cyan)', fontSize: '0.78rem', fontWeight: 600 }}>{p.readiness_score || 0}%</span>
                      </div>
                      <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>{p.institution}</div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%', marginTop: '0.25rem' }}>
                        <span className={`badge badge-${p.status}`}>{p.status}</span>
                        <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>Strength: {Math.round(p.relationship_strength * 100)}%</span>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Right: selected PI dossier details */}
            <div className="glass-panel detail-panel">
              {selectedProf ? (
                <div>
                  <div className="detail-header">
                    <div>
                      <h2 className="detail-title">{selectedProf.professor_name}</h2>
                      <p className="detail-subtitle">{selectedProf.institution}</p>
                    </div>
                    
                    {/* Status change controls and Deep Analysis trigger */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
                      {/* Lite Scan checkbox */}
                      <label style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.78rem', color: 'var(--text-secondary)', cursor: 'pointer', userSelect: 'none' }}>
                        <input
                          type="checkbox"
                          checked={liteScan}
                          onChange={e => setLiteScan(e.target.checked)}
                          style={{ accentColor: 'var(--accent-cyan)', width: 13, height: 13 }}
                        />
                        Lite Scan
                      </label>

                      {/* Deep / Lite run button */}
                      <button
                        className="btn-primary"
                        onClick={() => triggerDeepAnalysis(selectedProf.professor_name, liteScan)}
                        style={{ padding: '0.35rem 0.75rem', fontSize: '0.78rem', display: 'flex', alignItems: 'center', gap: '0.35rem', background: liteScan ? 'rgba(0,230,230,0.12)' : undefined, borderColor: liteScan ? 'var(--accent-cyan)' : undefined }}
                      >
                        <RefreshCw size={13} />
                        {liteScan ? 'Run Lite Scan' : 'Run Deep Analysis'}
                      </button>

                      {/* Rescan button (re-run using cached papers) */}
                      <button
                        className="btn-secondary"
                        title="Rescan: update questions using cached paper analyses, no new PDF downloads"
                        onClick={() => triggerDeepAnalysis(selectedProf.professor_name, true)}
                        style={{ padding: '0.35rem 0.6rem', fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}
                      >
                        <RefreshCw size={12} /> Rescan
                      </button>

                      <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Stage:</span>
                      <select
                        className="form-control"
                        value={selectedProf.status}
                        onChange={(e) => handleStatusChange(selectedProf.openalex_id || selectedProf.professor_name, e.target.value)}
                        style={{ padding: '0.35rem 0.5rem', background: 'var(--bg-surface)' }}
                      >
                        <option value="discovered">Discovered</option>
                        <option value="contacted">Contacted</option>
                        <option value="in_conversation">In Conversation</option>
                        <option value="mentor">Mentor</option>
                        <option value="resolved">Resolved</option>
                        <option value="internship_candidate">Internship Candidate</option>
                        <option value="dormant">Dormant</option>
                        <option value="ignored">Ignored</option>
                      </select>
                    </div>
                  </div>

                  {/* Academic Metrics Dossier Grid */}
                  <div className="grid-4col" style={{ marginBottom: '1.5rem', background: 'rgba(255,255,255,0.015)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                    <div>
                      <div style={{ fontSize: '0.68rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px', fontWeight: 600 }}>Career Stage</div>
                      <div style={{ fontSize: '0.88rem', fontWeight: 700, color: '#fff', marginTop: '0.2rem', textTransform: 'capitalize' }}>
                        {selectedProf.career_stage || 'Unknown'}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '0.68rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px', fontWeight: 600 }}>H-Index</div>
                      <div style={{ fontSize: '0.88rem', fontWeight: 700, color: '#fff', marginTop: '0.2rem' }}>
                        {selectedProf.h_index || 'N/A'}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '0.68rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px', fontWeight: 600 }}>Last Pub Year</div>
                      <div style={{ fontSize: '0.88rem', fontWeight: 700, color: '#fff', marginTop: '0.2rem' }}>
                        {selectedProf.last_publication_year || 'N/A'}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '0.68rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px', fontWeight: 600 }}>Supervision Signal</div>
                      <div style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--accent-emerald)', marginTop: '0.2rem' }}>
                        {selectedProf.student_supervision_signal ? `${Math.round(selectedProf.student_supervision_signal * 100)}%` : '0%'}
                      </div>
                    </div>
                  </div>

                  {/* Explanations Tooltip KPI row */}
                  <div className="grid-3col" style={{ marginBottom: '1.5rem' }}>
                    <div className="kpi-card" style={{ cursor: 'help' }} onClick={() => { setScoreExplProf(selectedProf); setScoreModalOpen(true); }}>
                      <div className="kpi-val" style={{ color: 'var(--accent-amber)' }}>{selectedProf.readiness_score || 0}%</div>
                      <div className="kpi-lbl">Readiness Score <span style={{ fontSize: '0.65rem', color: 'var(--accent-cyan)' }}>Explain</span></div>
                    </div>
                    <div className="kpi-card">
                      <div className="kpi-val" style={{ color: 'var(--accent-violet)' }}>{Math.round(selectedProf.relationship_strength * 100)}%</div>
                      <div className="kpi-lbl">Relationship Strength</div>
                    </div>
                    <div className="kpi-card">
                      <div className="kpi-val" style={{ color: 'var(--accent-emerald)' }}>{selectedProf.accessibility_score || 0}</div>
                      <div className="kpi-lbl">Accessibility</div>
                    </div>
                  </div>

                  {/* Relationship Memory card & Warning Checklist */}
                  <div className="grid-2col" style={{ marginBottom: '1.5rem' }}>
                    <div className="glass-panel" style={{ background: 'rgba(255,255,255,0.015)', padding: '1.25rem' }}>
                      <div className="panel-title" style={{ fontSize: '0.95rem' }}><Database size={16} /> Relationship Memory</div>
                      <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', whiteSpace: 'pre-line', lineHeight: '1.5', marginTop: '0.5rem' }}>
                        {selectedProf.relationship_summary}
                      </p>
                    </div>

                    <div>
                      {/* Avoid Repeating checklist box */}
                      <div className="warning-box">
                        <div style={{ fontWeight: 600, color: '#fda4af', fontSize: '0.8rem', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                          <AlertTriangle size={14} color="#fda4af" /> Avoid Repeating
                        </div>
                        {selectedProf.avoid_repeating && selectedProf.avoid_repeating.map((warn, idx) => (
                          <div key={idx} className="warning-item" style={{ marginTop: '0.25rem' }}>
                            <Check size={12} color="#fda4af" /> {warn}
                          </div>
                        ))}
                      </div>

                      {/* Log Professor Reply Box */}
                      <div className="glass-panel" style={{ background: 'rgba(255,255,255,0.01)', padding: '1rem' }}>
                        <div className="panel-title" style={{ fontSize: '0.88rem' }}><Mail size={16} /> Process Professor Reply</div>
                        <textarea
                          placeholder="Paste response email body to analyze suggestions and extract tasks..."
                          value={replyText}
                          onChange={(e) => setReplyText(e.target.value)}
                          className="form-control"
                          rows={3}
                          style={{ fontSize: '0.8rem', marginTop: '0.5rem', resize: 'vertical' }}
                        />
                        <button 
                          className="btn-primary" 
                          disabled={!replyText.trim() || isExtracting}
                          onClick={() => {
                            // Find the latest active interaction for this professor to associate the reply with
                            fetch(`/api/crm/interactions?prof_id=${encodeURIComponent(selectedProf.openalex_id || selectedProf.professor_name)}`)
                              .then(r => r.json())
                              .then(ints => {
                                const activeInt = ints.find(i => i.status === 'sent');
                                if (activeInt) {
                                  triggerReplyExtraction(activeInt);
                                } else {
                                  // Fallback to latest drafted or mock interaction
                                  triggerReplyExtraction(ints[ints.length - 1] || { id: 'dummy', investigation_id: '', blocker_id: '' });
                                }
                              });
                          }}
                          style={{ marginTop: '0.5rem', width: '100%', fontSize: '0.78rem' }}
                        >
                          {isExtracting ? 'Extracting suggestions...' : 'Log & Analyze Reply'}
                        </button>
                      </div>
                    </div>
                  </div>

                  {/* Suggested Outreach Messages / Tension bridges */}
                  <div style={{ marginBottom: '1.5rem' }}>
                    <div className="panel-title" style={{ fontSize: '0.95rem' }}><Send size={16} /> Suggested outreach pitch (Tars Studio)</div>
                    <div className="stack-item" style={{ marginTop: '0.5rem', borderLeft: '3px solid var(--accent-violet)' }}>
                      <strong style={{ fontSize: '0.85rem', color: '#fff' }}>Suggested Question:</strong>
                      <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', fontStyle: 'italic', margin: '0.25rem 0' }}>
                        "Would your curriculum learning algorithms hold up under representation drift or CKA degradation constraints?"
                      </p>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                        Relates to Blocker: Empirical dataset comparing representation geometry under curriculum learning.
                      </div>
                    </div>
                  </div>

                  {/* Research Insights: Questions Worth Asking, Analyzed Papers, Contradictions */}
                  <div className="grid-2col" style={{ marginBottom: '1.5rem' }}>
                    {/* Questions Worth Asking — Versioned Outreach Drawer */}
                    <div className="glass-panel" style={{ background: 'rgba(255,255,255,0.015)', padding: '1.25rem' }}>
                      <div className="panel-title" style={{ fontSize: '0.92rem', display: 'flex', alignItems: 'center', gap: '0.5rem', justifyContent: 'space-between' }}>
                        <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                          <HelpCircle size={16} color="var(--accent-cyan)" /> Outreach Pitch History
                        </span>
                        {/* Version selector */}
                        {selectedProf.questions_worth_asking && selectedProf.questions_worth_asking.length > 0 && (
                          <select
                            value={selectedQuestionIdx}
                            onChange={e => setSelectedQuestionIdx(Number(e.target.value))}
                            className="form-control"
                            style={{ padding: '0.2rem 0.4rem', fontSize: '0.72rem', maxWidth: 130 }}
                          >
                            {selectedProf.questions_worth_asking.map((q, idx) => (
                              <option key={idx} value={idx}>
                                v{idx + 1} — {typeof q === 'object' ? (q.status || 'generated') : 'generated'}
                              </option>
                            ))}
                          </select>
                        )}
                      </div>

                      {selectedProf.questions_worth_asking && selectedProf.questions_worth_asking.length > 0 ? (() => {
                        const raw = selectedProf.questions_worth_asking[Math.min(selectedQuestionIdx, selectedProf.questions_worth_asking.length - 1)];
                        const entry = typeof raw === 'string' ? { question: raw, email_draft: '', self_context: '', paper_referenced: '', status: 'generated' } : raw;
                        return (
                          <div style={{ marginTop: '0.75rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                            {/* Paper referenced */}
                            {entry.paper_referenced && (
                              <div style={{ fontSize: '0.72rem', color: 'var(--accent-cyan)', borderLeft: '2px solid var(--accent-cyan)', paddingLeft: '0.5rem' }}>
                                📄 Based on: <em>{entry.paper_referenced}</em>
                              </div>
                            )}

                            {/* Intellectual question */}
                            <div style={{ background: 'rgba(0,230,230,0.06)', borderRadius: 6, padding: '0.6rem 0.75rem', borderLeft: '3px solid var(--accent-cyan)' }}>
                              <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '0.2rem' }}>Intellectual Question</div>
                              <div style={{ fontSize: '0.82rem', color: '#fff', fontStyle: 'italic' }}>{entry.question}</div>
                            </div>

                            {/* Email draft panel with copy button */}
                            {entry.email_draft && (
                              <div style={{ background: 'rgba(255,255,255,0.02)', borderRadius: 6, padding: '0.75rem', border: '1px solid var(--border-color)' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                                  <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>📧 Email Draft</div>
                                  <button
                                    className="btn-secondary"
                                    style={{ padding: '0.15rem 0.5rem', fontSize: '0.68rem' }}
                                    onClick={() => navigator.clipboard.writeText(entry.email_draft)}
                                  >
                                    Copy Email
                                  </button>
                                </div>
                                <pre style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', whiteSpace: 'pre-wrap', margin: 0, fontFamily: 'inherit', lineHeight: 1.55 }}>{entry.email_draft}</pre>
                              </div>
                            )}

                            {/* Private self-context — visible only, excluded from copy */}
                            {entry.self_context && (
                              <div style={{ background: 'rgba(251,146,60,0.07)', borderRadius: 6, padding: '0.6rem 0.75rem', border: '1px dashed rgba(251,146,60,0.35)' }}>
                                <div style={{ fontSize: '0.68rem', color: '#fb923c', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '0.25rem' }}>🔒 Private Context (not sent to professor)</div>
                                <div style={{ fontSize: '0.78rem', color: 'rgba(251,146,60,0.85)', fontStyle: 'italic', lineHeight: 1.5 }}>{entry.self_context}</div>
                              </div>
                            )}

                            {/* Status selector */}
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Status:</span>
                              <select
                                value={entry.status || 'generated'}
                                onChange={async e => {
                                  const profId = selectedProf.openalex_id || selectedProf.professor_name;
                                  await fetch(`/api/crm/professors/${encodeURIComponent(profId)}/update_question_status`, {
                                    method: 'PUT',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({ question_index: selectedQuestionIdx, status: e.target.value })
                                  });
                                  loadProfessors(profId);
                                }}
                                className="form-control"
                                style={{ padding: '0.2rem 0.4rem', fontSize: '0.72rem', maxWidth: 130 }}
                              >
                                <option value="generated">Generated</option>
                                <option value="edited">Edited</option>
                                <option value="sent">Sent</option>
                                <option value="answered">Answered</option>
                                <option value="archived">Archived</option>
                              </select>
                            </div>
                          </div>
                        );
                      })() : (
                        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontStyle: 'italic', marginTop: '0.5rem' }}>
                          No outreach pitches yet. Run deep analysis or lite scan to generate questions.
                        </div>
                      )}
                    </div>

                    {/* Indexed/Analyzed Papers */}
                    <div className="glass-panel" style={{ background: 'rgba(255,255,255,0.015)', padding: '1.25rem' }}>
                      <div className="panel-title" style={{ fontSize: '0.92rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}><FileText size={16} color="var(--accent-violet)" /> Indexed Publications</div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem', maxHeight: '200px', overflowY: 'auto' }}>
                        {selectedProf.papers_at_last_check && selectedProf.papers_at_last_check.length > 0 ? (
                          selectedProf.papers_at_last_check.map((p, idx) => (
                            <div key={idx} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.5rem', fontSize: '0.78rem', borderBottom: '1px solid rgba(255,255,255,0.03)', paddingBottom: '0.35rem' }}>
                              <span style={{ color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
                                {p.startsWith('http') ? p.split('/').slice(-2).join('/') : p}
                              </span>
                              {p.startsWith('10.') || p.includes('/') ? (
                                <a href={p.startsWith('http') ? p : `https://doi.org/${p}`} target="_blank" rel="noopener noreferrer" style={{ color: 'var(--accent-cyan)', display: 'inline-flex', alignItems: 'center' }}>
                                  <ExternalLink size={12} />
                                </a>
                              ) : null}
                            </div>
                          ))
                        ) : (
                          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                            No papers indexed yet. Run deep analysis to find publications.
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* PI Research Dossier Panel */}
                  <div className="glass-panel" style={{ padding: '1.25rem', marginBottom: '1.5rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.6rem', marginBottom: '0.8rem' }}>
                      <div className="panel-title" style={{ fontSize: '1rem', margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <Database size={18} color="var(--accent-emerald)" />
                        PI Research Dossier (Deep Insights)
                      </div>
                      
                      {profProfile && (
                        <div style={{ display: 'flex', gap: '0.4rem' }}>
                          {['synthesis', 'papers', 'coauthors'].map(tab => (
                            <button
                              key={tab}
                              className={`sub-tab-btn ${dossierTab === tab ? 'active' : ''}`}
                              onClick={() => setDossierTab(tab)}
                              style={{ padding: '0.25rem 0.6rem', fontSize: '0.75rem', borderRadius: 4 }}
                            >
                              {tab === 'synthesis' ? 'Holistic Synthesis' : tab === 'papers' ? 'Paper Analyses' : 'Coauthors'}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>

                    {!profProfile ? (
                      <div style={{ padding: '1.5rem 0', textAlign: 'center' }}>
                        <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                          Detailed research dossier and deep LLM paper analyses not loaded.
                        </div>
                        <button
                          className="btn-secondary"
                          style={{ fontSize: '0.75rem', padding: '0.3rem 0.6rem' }}
                          onClick={() => triggerDeepAnalysis(selectedProf.professor_name, liteScan)}
                        >
                          Run Analysis to Build Dossier
                        </button>
                      </div>
                    ) : (
                      <div>
                        {/* Tab 1: Holistic Synthesis */}
                        {dossierTab === 'synthesis' && (
                          <div style={{ fontSize: '0.85rem', color: 'var(--text-primary)', lineHeight: 1.6, whiteSpace: 'pre-wrap', maxHeight: '400px', overflowY: 'auto', paddingRight: '0.5rem' }}>
                            {profProfile.research_synthesis || "No synthesis report generated. Run a deep analysis to create."}
                          </div>
                        )}

                        {/* Tab 2: Deep Paper Analyses */}
                        {dossierTab === 'papers' && (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', maxHeight: '500px', overflowY: 'auto', paddingRight: '0.5rem' }}>
                            {profProfile.publications && profProfile.publications.length > 0 ? (
                              profProfile.publications.map((pub, idx) => {
                                const hasAnalysis = pub.analysis && (pub.analysis.summary || pub.analysis.methodology || pub.analysis.key_contributions);
                                const isExpanded = expandedPaperIdx === idx;
                                return (
                                  <div 
                                    key={idx} 
                                    className="stack-item" 
                                    style={{ 
                                      background: isExpanded ? 'rgba(255,255,255,0.015)' : 'rgba(255,255,255,0.005)',
                                      borderLeft: hasAnalysis ? '3px solid var(--accent-emerald)' : '3px solid var(--border-color)',
                                      padding: '0.75rem',
                                      transition: 'all 0.2s ease'
                                    }}
                                  >
                                    <div 
                                      style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', cursor: 'pointer' }}
                                      onClick={() => setExpandedPaperIdx(isExpanded ? null : idx)}
                                    >
                                      <div style={{ flex: 1, paddingRight: '0.5rem' }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flexWrap: 'wrap' }}>
                                          <span className="badge badge-in_conversation" style={{ fontSize: '0.68rem', padding: '0.1rem 0.35rem', background: 'rgba(16,185,129,0.1)', color: 'var(--accent-emerald)' }}>
                                            {pub.year || 'N/A'}
                                          </span>
                                          <strong style={{ color: '#fff', fontSize: '0.88rem' }}>{pub.title}</strong>
                                        </div>
                                        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                                          {pub.venue || 'Unknown Venue'} | Citations: {pub.citations_max || pub.citations_openalex || 0}
                                        </div>
                                      </div>
                                      <span style={{ fontSize: '0.75rem', color: 'var(--accent-cyan)' }}>
                                        {isExpanded ? 'Collapse' : 'Expand'}
                                      </span>
                                    </div>

                                    {isExpanded && (
                                      <div style={{ marginTop: '0.75rem', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '0.6rem', fontSize: '0.85rem', display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                                        {pub.abstract && (
                                          <div>
                                            <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '0.15rem' }}>Abstract</div>
                                            <div style={{ color: 'var(--text-secondary)', fontStyle: 'italic', lineHeight: 1.45 }}>{pub.abstract}</div>
                                          </div>
                                        )}

                                        {hasAnalysis ? (
                                          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', background: 'rgba(0,230,230,0.02)', padding: '0.6rem', borderRadius: 6, border: '1px solid rgba(0,230,230,0.08)' }}>
                                            {pub.analysis.summary && (
                                              <div>
                                                <div style={{ fontSize: '0.68rem', color: 'var(--accent-cyan)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '0.15rem' }}>Research Summary</div>
                                                <div style={{ color: 'var(--text-secondary)', lineHeight: 1.45 }}>{pub.analysis.summary}</div>
                                              </div>
                                            )}
                                            {pub.analysis.key_contributions && (
                                              <div>
                                                <div style={{ fontSize: '0.68rem', color: 'var(--accent-cyan)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '0.15rem' }}>Key Contributions</div>
                                                <div style={{ color: 'var(--text-secondary)', lineHeight: 1.45 }}>
                                                  {typeof pub.analysis.key_contributions === 'string' ? pub.analysis.key_contributions : (
                                                    <ul style={{ margin: '0.2rem 0 0 1rem', padding: 0 }}>
                                                      {pub.analysis.key_contributions.map((c, i) => <li key={i}>{c}</li>)}
                                                    </ul>
                                                  )}
                                                </div>
                                              </div>
                                            )}
                                            {pub.analysis.methodology && (
                                              <div>
                                                <div style={{ fontSize: '0.68rem', color: 'var(--accent-cyan)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '0.15rem' }}>Methodology</div>
                                                <div style={{ color: 'var(--text-secondary)', lineHeight: 1.45 }}>{pub.analysis.methodology}</div>
                                              </div>
                                            )}
                                            {pub.analysis.datasets_used && (
                                              <div>
                                                <div style={{ fontSize: '0.68rem', color: 'var(--accent-cyan)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '0.15rem' }}>Datasets Used</div>
                                                <div style={{ color: 'var(--text-secondary)', lineHeight: 1.45 }}>{pub.analysis.datasets_used}</div>
                                              </div>
                                            )}
                                            {pub.analysis.limitations && (
                                              <div>
                                                <div style={{ fontSize: '0.68rem', color: 'var(--accent-rose)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '0.15rem' }}>Limitations</div>
                                                <div style={{ color: 'var(--text-secondary)', lineHeight: 1.45 }}>{pub.analysis.limitations}</div>
                                              </div>
                                            )}
                                          </div>
                                        ) : (
                                          <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                                            No deep NLI/relevance analysis for this paper. (Lite scan processes abstract metadata only).
                                          </div>
                                        )}

                                        {pub.doi && (
                                          <div style={{ display: 'flex', gap: '0.4rem', marginTop: '0.2rem' }}>
                                            <a 
                                              href={`https://doi.org/${pub.doi}`} 
                                              target="_blank" 
                                              rel="noopener noreferrer" 
                                              className="btn-secondary" 
                                              style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', padding: '0.15rem 0.45rem', fontSize: '0.68rem' }}
                                            >
                                              <ExternalLink size={10} /> DOI Link
                                            </a>
                                            {pub.oa_url && (
                                              <a 
                                                href={pub.oa_url} 
                                                target="_blank" 
                                                rel="noopener noreferrer" 
                                                className="btn-secondary" 
                                                style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', padding: '0.15rem 0.45rem', fontSize: '0.68rem' }}
                                              >
                                                <ExternalLink size={10} /> Open Access URL
                                              </a>
                                            )}
                                          </div>
                                        )}
                                      </div>
                                    )}
                                  </div>
                                );
                              })
                            ) : (
                              <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontStyle: 'italic' }}>No publications found.</div>
                            )}
                          </div>
                        )}

                        {/* Tab 3: Coauthors */}
                        {dossierTab === 'coauthors' && (
                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '0.6rem', maxHeight: '400px', overflowY: 'auto', paddingRight: '0.5rem' }}>
                            {profProfile.coauthors && profProfile.coauthors.length > 0 ? (
                              profProfile.coauthors.map((author, idx) => (
                                <div key={idx} className="stack-item" style={{ background: 'rgba(255,255,255,0.01)', padding: '0.5rem 0.6rem' }}>
                                  <strong style={{ color: '#fff', fontSize: '0.82rem' }}>{author.name || author.display_name}</strong>
                                  <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginTop: '0.15rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                    {author.affiliation || 'No known affiliation'}
                                  </div>
                                  <div style={{ fontSize: '0.68rem', color: 'var(--accent-cyan)', marginTop: '0.2rem' }}>
                                    Coauthored Works: {author.coauthored_papers || author.works_count || 1}
                                  </div>
                                </div>
                              ))
                            ) : (
                              <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontStyle: 'italic', gridColumn: '1/-1' }}>No coauthors listed.</div>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Contradictions Introduced */}
                  {selectedProf.contradictions_introduced && selectedProf.contradictions_introduced.length > 0 && (
                    <div style={{ marginBottom: '1.5rem' }}>
                      <div className="panel-title" style={{ fontSize: '0.95rem', color: 'var(--accent-rose)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}><AlertCircle size={16} /> Contradictions/Theoretical Tensions Introduced</div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem' }}>
                        {selectedProf.contradictions_introduced.map((c, idx) => (
                          <div key={idx} className="stack-item" style={{ background: 'rgba(244,63,94,0.03)', borderLeft: '3px solid var(--accent-rose)', color: '#fda4af', padding: '0.6rem' }}>
                            <p style={{ fontSize: '0.82rem' }}>{c}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Conversation history timeline */}
                  <div>
                    <div className="panel-title" style={{ fontSize: '0.95rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}><Calendar size={16} /> Interaction Timeline</div>
                    <div className="timeline" style={{ marginTop: '1rem' }}>
                      {/* Timeline content is fetched or represented by events */}
                      {selectedProf.events && selectedProf.events.length > 0 ? (
                        selectedProf.events.map((ev, idx) => (
                          <div key={idx} className="timeline-event">
                            <div className="timeline-node"></div>
                            <div className="timeline-event-header">
                              <span style={{ textTransform: 'uppercase', fontWeight: 700, color: 'var(--accent-violet)' }}>{ev.event_type}</span>
                            </div>
                            <p style={{ fontSize: '0.8rem', color: 'var(--text-primary)' }}>{ev.notes}</p>
                          </div>
                        ))
                      ) : (
                        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>No historical timeline events yet.</div>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="empty-state">
                  <h3>Select a relationship dossier</h3>
                  <p>Choose a profiled researcher from the sidebar to inspect timeline logs and memory summaries.</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* PIPELINE KANBAN TAB */}
        {activeTab === 'Pipeline' && (
          <div className="kanban-board">
            {[
              { title: 'Interested', statusCodes: ['discovered', 'drafted'] },
              { title: 'Contacted', statusCodes: ['contacted'] },
              { title: 'Replied', statusCodes: ['replied'] },
              { title: 'Ongoing Conversation', statusCodes: ['in_conversation'] },
              { title: 'Mentor', statusCodes: ['mentor', 'resolved'] },
              { title: 'Internship Candidate', statusCodes: ['internship_candidate', 'internship'] }
            ].map((col, cidx) => {
              const cards = professors.filter(p => col.statusCodes.includes(p.status));
              return (
                <div key={cidx} className="kanban-col">
                  <div className="kanban-header">
                    <span>{col.title}</span>
                    <span style={{ color: 'var(--text-muted)' }}>{cards.length}</span>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.50rem', overflowY: 'auto', flex: 1 }}>
                    {cards.length === 0 ? (
                      <div style={{ textAlign: 'center', padding: '1rem 0', color: 'var(--text-muted)', fontSize: '0.75rem' }}>Empty Stage</div>
                    ) : (
                      cards.map((p, pidx) => (
                        <div key={pidx} className="kanban-card">
                          <div className="kanban-card-title">{p.professor_name}</div>
                          <div className="kanban-card-sub">{p.institution}</div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.72rem', color: 'var(--accent-cyan)' }}>
                            <span>Readiness: {p.readiness_score || 0}%</span>
                            <span style={{ color: 'var(--text-muted)' }}>Str: {Math.round(p.relationship_strength * 100)}%</span>
                          </div>
                          
                          {/* Column quick move actions */}
                          <div style={{ display: 'flex', gap: '0.2rem', marginTop: '0.25rem', justifyContent: 'flex-end' }}>
                            <select 
                              className="form-control"
                              value={p.status}
                              onChange={(e) => movePIStatus(p.openalex_id || p.professor_name, e.target.value)}
                              style={{ width: '100%', padding: '0.15rem 0.35rem', fontSize: '0.68rem', height: '22px' }}
                            >
                              <option value="discovered">Interested</option>
                              <option value="contacted">Contacted</option>
                              <option value="in_conversation">Ongoing</option>
                              <option value="mentor">Mentor</option>
                              <option value="internship_candidate">Internship Candidate</option>
                              <option value="dormant">Dormant</option>
                            </select>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* UNIVERSITIES TAB */}
        {activeTab === 'Universities' && (
          <div className="sidebar-layout">
            {/* Sidebar of Universities */}
            <div className="glass-panel sidebar-panel">
              <div className="panel-title">Discovery Coverage</div>
              <div className="sidebar-list">
                {universities.map((uni, idx) => {
                  const isActive = selectedUni && selectedUni.name === uni.name;
                  return (
                    <button 
                      key={idx}
                      className={`list-item-btn ${isActive ? 'active' : ''}`}
                      onClick={() => setSelectedUni(uni)}
                    >
                      <strong style={{ color: '#fff', fontSize: '0.88rem' }}>{uni.alias}</strong>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', marginTop: '0.2rem' }}>
                        <span>Researchers: {uni.total_researchers || uni.total_professors || 100}</span>
                        <span style={{ color: 'var(--accent-emerald)', fontWeight: 600 }}>Coverage: {uni.relevant_coverage || uni.coverage || 0}%</span>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* University Coverage panel */}
            <div className="glass-panel detail-panel">
              {selectedUni ? (
                <div>
                  <div className="detail-header">
                    <div>
                      <h2 className="detail-title">{selectedUni.name}</h2>
                      <p className="detail-subtitle">Active Outreach Opportunities</p>
                    </div>

                    {/* University Scanning Buttons */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
                      {/* Lite Scan checkbox */}
                      <label style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.78rem', color: 'var(--text-secondary)', cursor: 'pointer', userSelect: 'none' }}>
                        <input
                          type="checkbox"
                          checked={liteScan}
                          onChange={e => setLiteScan(e.target.checked)}
                          style={{ accentColor: 'var(--accent-cyan)', width: 13, height: 13 }}
                        />
                        Lite Scan
                      </label>

                      {/* Deep / Lite run button */}
                      <button
                        className="btn-primary"
                        onClick={() => triggerUniversityDiscovery(selectedUni.name, liteScan)}
                        style={{ padding: '0.35rem 0.75rem', fontSize: '0.78rem', display: 'flex', alignItems: 'center', gap: '0.35rem', background: liteScan ? 'rgba(0,230,230,0.12)' : undefined, borderColor: liteScan ? 'var(--accent-cyan)' : undefined }}
                      >
                        <RefreshCw size={13} />
                        {liteScan ? 'Run Lite Scan' : 'Run Deep Analysis'}
                      </button>

                      {/* Rescan button (re-run using cache) */}
                      <button
                        className="btn-secondary"
                        title="Rescan: update university profile using cached paper analyses, no new downloads"
                        onClick={() => triggerUniversityDiscovery(selectedUni.name, true)}
                        style={{ padding: '0.35rem 0.6rem', fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}
                      >
                        <RefreshCw size={12} /> Rescan
                      </button>
                    </div>
                  </div>

                  {/* Coverage stats block */}
                  <div className="grid-2col" style={{ marginBottom: '1.5rem' }}>
                    <div className="glass-panel" style={{ background: 'rgba(255,255,255,0.015)', justifyContent: 'center', alignItems: 'center', padding: '1.5rem', textAlign: 'center' }}>
                      <div style={{ fontSize: '2.5rem', fontWeight: 800, color: 'var(--accent-emerald)', fontFamily: 'var(--font-display)' }}>{selectedUni.relevant_coverage || selectedUni.coverage || 0}%</div>
                      <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '0.25rem', fontWeight: 600 }}>Relevant Coverage</div>
                      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.5rem', maxWidth: '300px' }}>
                        Estimated coverage of key researchers in our focus domains (computational cognition, active inference, meta-science).
                      </div>
                    </div>

                    <div className="glass-panel" style={{ background: 'rgba(255,255,255,0.01)', padding: '1.25rem' }}>
                      <div className="panel-title" style={{ fontSize: '0.9rem' }}><MapPin size={16} /> Key Metrics</div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem', fontSize: '0.85rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: 'var(--text-secondary)' }}>Deep Scanned / Est. Relevant:</span>
                          <span style={{ color: '#fff', fontWeight: 700 }}>{selectedUni.deep_scanned} / {selectedUni.total_relevant_estimated}</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: 'var(--text-secondary)' }}>Outreach Rate (Contacted / Indexed):</span>
                          <span style={{ color: 'var(--accent-violet)', fontWeight: 700 }}>{selectedUni.contacted} / {selectedUni.indexed_professors} ({selectedUni.outreach_coverage || 0}%)</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: 'var(--text-secondary)' }}>Ongoing Discussions:</span>
                          <span style={{ color: 'var(--accent-emerald)', fontWeight: 700 }}>{selectedUni.ongoing}</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: 'var(--text-secondary)' }}>Replies Received:</span>
                          <span style={{ color: 'var(--accent-cyan)', fontWeight: 700 }}>{selectedUni.replied}</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Researcher Funnel section */}
                  <div style={{ marginBottom: '1.5rem' }}>
                    <div className="panel-title" style={{ fontSize: '0.95rem' }}><GitBranch size={16} /> Researcher Funnel</div>
                    <div className="funnel-flow" style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '0.75rem' }}>
                      {[
                        { label: 'Discovered', val: selectedUni.total_researchers || selectedUni.total_professors, desc: 'Total researchers with h-index > 5 crawled from OpenAlex', color: 'var(--text-secondary)' },
                        { label: 'Screened', val: selectedUni.screened || 0, desc: 'Researchers matching broad target focus concepts', color: 'var(--accent-amber)' },
                        { label: 'Relevant (Est.)', val: selectedUni.total_relevant_estimated || 1, desc: 'Estimated researchers matching active investigation blockers', color: 'var(--accent-violet)' },
                        { label: 'Deep Scanned', val: selectedUni.deep_scanned || 0, desc: 'Detailed profiles & program dossiers scanned', color: 'var(--accent-cyan)' },
                        { label: 'Contacted', val: selectedUni.contacted || 0, desc: 'Outreach emails sent to relevant candidates', color: 'var(--accent-rose)' },
                        { label: 'Replied', val: selectedUni.replied || 0, desc: 'Response emails received', color: 'var(--accent-amber)' },
                        { label: 'Active Conversation', val: selectedUni.ongoing || 0, desc: 'Ongoing collaboration / internship discussion', color: 'var(--accent-emerald)' }
                      ].map((stage, idx, arr) => {
                        const prevVal = idx > 0 ? arr[idx-1].val : null;
                        const conversion = prevVal && prevVal > 0 ? Math.round((stage.val / prevVal) * 100.0) : null;
                        return (
                          <div key={idx} style={{ display: 'flex', flexDirection: 'column' }}>
                            {idx > 0 && (
                              <div style={{ display: 'flex', alignItems: 'center', paddingLeft: '1.75rem', height: '1.25rem', color: 'var(--text-muted)', gap: '1rem' }}>
                                <div style={{ width: '2px', background: 'rgba(255,255,255,0.06)', height: '100%' }}></div>
                                {conversion !== null && (
                                  <span style={{ fontSize: '0.68rem', fontWeight: 600, color: 'var(--accent-cyan)', background: 'rgba(6,182,212,0.08)', padding: '0.05rem 0.35rem', borderRadius: '4px' }}>
                                    ↓ {conversion}% conversion
                                  </span>
                                )}
                              </div>
                            )}
                            <div style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid var(--border-color)', padding: '0.6rem 1rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderRadius: '8px' }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                                <div style={{ 
                                  width: '1.75rem', 
                                  height: '1.75rem', 
                                  borderRadius: '50%', 
                                  background: 'rgba(255,255,255,0.02)', 
                                  border: `1px solid ${stage.color}`, 
                                  display: 'flex', 
                                  alignItems: 'center', 
                                  justifyContent: 'center',
                                  fontWeight: 700,
                                  fontSize: '0.75rem',
                                  color: stage.color
                                }}>
                                  {idx + 1}
                                </div>
                                <div>
                                  <strong style={{ color: '#fff', fontSize: '0.85rem' }}>{stage.label}</strong>
                                  <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>{stage.desc}</div>
                                </div>
                              </div>
                              <div style={{ textAlign: 'right' }}>
                                <span style={{ fontSize: '1.15rem', fontWeight: 800, color: '#fff', fontFamily: 'var(--font-display)' }}>
                                  {stage.val.toLocaleString()}
                                </span>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Programs explorer list */}
                  <div style={{ marginBottom: '1.5rem' }}>
                    <div className="panel-title" style={{ fontSize: '0.95rem' }}><Compass size={16} /> Clustered Programs</div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '0.75rem' }}>
                      {/* Basic lookup of programs associated with this university */}
                      <div className="stack-item">
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <strong style={{ color: '#fff', fontSize: '0.95rem' }}>Representation Geometry Benchmark</strong>
                          <span className="badge badge-in_conversation">Active Fit</span>
                        </div>
                        <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.35rem' }}>
                          Investigates the alignment metrics (CKA/RSA) under curriculum constraints on neural representations.
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Indexed Professors section */}
                  <div>
                    <div className="panel-title" style={{ fontSize: '0.95rem' }}><UserCheck size={16} /> Indexed Professors ({professors.filter(p => p.institution === selectedUni.name).length})</div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.75rem' }}>
                      {professors.filter(p => p.institution === selectedUni.name).length === 0 ? (
                        <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontStyle: 'italic' }}>
                          No PIs indexed yet for this institution. Run a Lite Scan or Deep Analysis to discover.
                        </div>
                      ) : (
                        professors.filter(p => p.institution === selectedUni.name).map((prof, idx) => (
                          <div 
                            key={idx} 
                            className="stack-item" 
                            style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}
                            onClick={() => {
                              setSelectedProf(prof);
                              setActiveTab('Relationships');
                            }}
                          >
                            <div>
                              <strong style={{ color: '#fff', fontSize: '0.9rem' }}>{prof.professor_name}</strong>
                              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.15rem' }}>
                                Stage: <span style={{ color: 'var(--accent-cyan)' }}>{prof.status}</span> | H-Index: {prof.h_index || 'N/A'}
                              </div>
                            </div>
                            <ChevronRight size={16} color="var(--text-muted)" />
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="empty-state">
                  <h3>Select a University</h3>
                  <p>Choose an institution from the sidebar to review outreach coverage percentages and indexed programs.</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* WORKSPACE TAB */}
        {activeTab === 'Workspace' && (
          <div>
            {/* Sub navigation tabs */}
            <div className="sub-tabs">
              {['Investigations', 'Tension Studio', 'Graph', 'Self Model'].map(sub => (
                <button 
                  key={sub}
                  className={`sub-tab-btn ${workspaceSubTab === sub ? 'active' : ''}`}
                  onClick={() => setWorkspaceSubTab(sub)}
                >
                  {sub === 'Graph' ? 'Research Graph' : sub}
                </button>
              ))}
            </div>

            {/* Investigations View */}
            {workspaceSubTab === 'Investigations' && (
              <div className="grid-2col">
                {/* Left: list of streams */}
                <div className="glass-panel">
                  <div className="panel-title"><GitBranch size={16} /> Active Investigations</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '0.5rem' }}>
                    {investigations.map((inv, idx) => (
                      <div key={idx} className="stack-item" style={{ cursor: 'pointer', borderLeft: inv.status === 'blocked' ? '3px solid var(--accent-rose)' : '3px solid var(--accent-emerald)' }} onClick={() => setSelectedNode({ type: 'investigation', ...inv })}>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <strong style={{ color: '#fff', fontSize: '0.92rem' }}>{inv.title}</strong>
                          <span className={`badge badge-${inv.status}`}>{inv.status}</span>
                        </div>
                        <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>{inv.description}</div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Right: details from selected node */}
                <div className="glass-panel">
                  <div className="panel-title"><Info size={16} /> Investigation details</div>
                  {selectedNode && selectedNode.type === 'investigation' ? (
                    <div>
                      <h3 style={{ color: '#fff', fontFamily: 'var(--font-display)', fontSize: '1.25rem' }}>{selectedNode.title}</h3>
                      <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem', alignItems: 'center' }}>
                        <span className={`badge badge-${selectedNode.status}`}>{selectedNode.status}</span>
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>ID: {selectedNode.id.substring(0, 8)}...</span>
                      </div>
                      
                      <div style={{ marginTop: '1.5rem' }}>
                        <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: 600 }}>Open Blockers ({selectedNode.open_blockers ? selectedNode.open_blockers.length : 0})</label>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem' }}>
                          {selectedNode.open_blockers && selectedNode.open_blockers.map((b, bidx) => (
                            <div key={bidx} style={{ background: 'rgba(255,255,255,0.015)', border: '1px solid var(--border-color)', borderRadius: '6px', padding: '0.65rem' }}>
                              <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--accent-rose)' }}>Blocker: {b.description}</div>
                              <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>Severity: {b.severity} | Type: {b.blocker_type}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="empty-state">
                      <h3>No selection</h3>
                      <p>Click on any investigation from the list to view its open blockers and outcomes.</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Tension Studio View */}
            {workspaceSubTab === 'Tension Studio' && (
              <div className="glass-panel" style={{ maxWidth: '800px', margin: '0 auto', width: '100%' }}>
                <div className="panel-title">Tension Studio</div>
                <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: '1.25rem' }}>
                  Contrast unresolved blockers in your active investigations with the claims or publications of target PIs to identify intellectual tensions.
                </p>

                <div className="grid-2col">
                  <div className="form-group">
                    <label>Investigation Stream</label>
                    <select className="form-control" style={{ background: 'var(--bg-surface)' }}>
                      <option>Does curriculum ordering affect coherence in EarthOS?</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label>PI Blocker</label>
                    <select className="form-control" style={{ background: 'var(--bg-surface)' }}>
                      <option>Empirical dataset comparing representation geometry under different curriculum orderings</option>
                    </select>
                  </div>
                </div>

                <div className="grid-2col">
                  <div className="form-group">
                    <label>Target Professor</label>
                    <select className="form-control" style={{ background: 'var(--bg-surface)' }}>
                      <option>Jordan W. Squair</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label>Suggested Message Topic</label>
                    <select className="form-control" style={{ background: 'var(--bg-surface)' }}>
                      <option>Validate representation alignment under curriculum ordering constraints</option>
                    </select>
                  </div>
                </div>

                <button className="btn-primary" style={{ marginTop: '0.5rem', alignSelf: 'flex-start' }}>
                  Analyze Intellectual Tension
                </button>
              </div>
            )}

            {/* Research Graph View */}
            {workspaceSubTab === 'Graph' && (
              <div id="graph-view-wrapper">
                <svg ref={graphRef} id="graph-canvas"></svg>
                
                {/* Node details sidebar */}
                {selectedNode && (
                  <div className="graph-sidebar">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '0.5rem', marginBottom: '0.75rem' }}>
                      <span style={{ fontSize: '0.7rem', color: 'var(--accent-violet)', textTransform: 'uppercase', fontWeight: 700 }}>{selectedNode.type}</span>
                      <button style={{ background: 'transparent', border: 'none', color: '#fff', fontSize: '1.2rem', cursor: 'pointer' }} onClick={() => setSelectedNode(null)}>&times;</button>
                    </div>
                    <h3 style={{ color: '#fff', fontSize: '1.1rem', fontFamily: 'var(--font-display)', fontWeight: 700 }}>{selectedNode.label}</h3>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.5rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                      {selectedNode.type === 'professor' && (
                        <>
                          <div><strong>Institution:</strong> {selectedNode.details?.university || 'N/A'}</div>
                          <div><strong>H-Index:</strong> {selectedNode.details?.h_index || 'N/A'}</div>
                        </>
                      )}
                      {selectedNode.type === 'program' && (
                        <>
                          <div><strong>PI:</strong> {selectedNode.details?.professor_name}</div>
                          <div><strong>University:</strong> {selectedNode.details?.university}</div>
                          <div><strong>Summary:</strong> {selectedNode.details?.summary}</div>
                        </>
                      )}
                      {selectedNode.type === 'investigation' && (
                        <>
                          <div><strong>Description:</strong> {selectedNode.details?.description || selectedNode.description}</div>
                          <div><strong>Status:</strong> {selectedNode.details?.status || selectedNode.status}</div>
                        </>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Self Model View */}
            {workspaceSubTab === 'Self Model' && (
              <div className="grid-2col">
                <div className="glass-panel">
                  <div className="panel-title"><Award size={16} /> Extracted Claims</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem' }}>
                    {selfModel.strong_claims && selfModel.strong_claims.map((claim, idx) => (
                      <div key={idx} className="stack-item">
                        <div style={{ fontSize: '0.85rem', color: '#fff', fontWeight: 600 }}>{claim.text}</div>
                        <div style={{ fontSize: '0.7rem', color: 'var(--accent-cyan)', marginTop: '0.25rem' }}>Confidence: {Math.round(claim.confidence * 100)}% | Domain: {claim.domain}</div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="glass-panel">
                  <div className="panel-title"><HelpCircle size={16} /> Open Questions</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem' }}>
                    {selfModel.open_questions && selfModel.open_questions.map((q, idx) => (
                      <div key={idx} className="stack-item">
                        <div style={{ fontSize: '0.85rem', color: '#fff' }}>{q.text}</div>
                        <div style={{ fontSize: '0.7rem', color: 'var(--accent-violet)', marginTop: '0.25rem' }}>Confidence: {Math.round(q.confidence * 100)}% | Type: {q.question_type}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      {/* CONFIRM SUGGESTED TASKS MODAL */}
      {extractionModalOpen && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div className="modal-header">
              <h3 style={{ color: '#fff', fontFamily: 'var(--font-display)', fontWeight: 700 }}>Confirm Relationship Intelligence Extract</h3>
              <button style={{ background: 'transparent', border: 'none', color: '#fff', fontSize: '1.5rem', cursor: 'pointer' }} onClick={() => setExtractionModalOpen(false)}>&times;</button>
            </div>
            
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1.25rem' }}>
              The local LLM engine parsed the email reply. Review the extracted suggestions and confirm which tasks to append to your active research checklist.
            </p>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {/* Paper suggestion checkbox */}
              {extractedIntel.suggested_paper && (
                <div style={{ background: 'rgba(255,255,255,0.015)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '1rem' }}>
                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.75rem' }}>
                    <input 
                      type="checkbox" 
                      id="confirm-paper-checkbox" 
                      checked={paperChecked} 
                      onChange={(e) => setPaperChecked(e.target.checked)} 
                      style={{ marginTop: '0.25rem', transform: 'scale(1.2)' }}
                    />
                    <div style={{ flex: 1 }}>
                      <label htmlFor="confirm-paper-checkbox" style={{ color: '#fff', fontWeight: 600, fontSize: '0.88rem', cursor: 'pointer' }}>
                        Create Read Paper Task:
                      </label>
                      <input 
                        type="text" 
                        value={extractedIntel.suggested_paper} 
                        onChange={(e) => setExtractedIntel({ ...extractedIntel, suggested_paper: e.target.value })}
                        className="form-control" 
                        style={{ width: '100%', marginTop: '0.35rem', fontSize: '0.8rem' }}
                      />
                      <div style={{ fontSize: '0.72rem', color: 'var(--accent-cyan)', marginTop: '0.25rem' }}>Confidence Score: {Math.round(extractedIntel.paper_confidence * 100)}%</div>
                    </div>
                  </div>
                </div>
              )}

              {/* Experiment suggestion checkbox */}
              {extractedIntel.suggested_experiment && (
                <div style={{ background: 'rgba(255,255,255,0.015)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '1rem' }}>
                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.75rem' }}>
                    <input 
                      type="checkbox" 
                      id="confirm-exp-checkbox" 
                      checked={experimentChecked} 
                      onChange={(e) => setExperimentChecked(e.target.checked)} 
                      style={{ marginTop: '0.25rem', transform: 'scale(1.2)' }}
                    />
                    <div style={{ flex: 1 }}>
                      <label htmlFor="confirm-exp-checkbox" style={{ color: '#fff', fontWeight: 600, fontSize: '0.88rem', cursor: 'pointer' }}>
                        Create Test Experiment Task:
                      </label>
                      <textarea 
                        value={extractedIntel.suggested_experiment} 
                        onChange={(e) => setExtractedIntel({ ...extractedIntel, suggested_experiment: e.target.value })}
                        className="form-control" 
                        rows={2}
                        style={{ width: '100%', marginTop: '0.35rem', fontSize: '0.8rem', resize: 'vertical' }}
                      />
                      <div style={{ fontSize: '0.72rem', color: 'var(--accent-violet)', marginTop: '0.25rem' }}>Confidence Score: {Math.round(extractedIntel.experiment_confidence * 100)}%</div>
                    </div>
                  </div>
                </div>
              )}

              {(!extractedIntel.suggested_paper && !extractedIntel.suggested_experiment) && (
                <div style={{ textAlign: 'center', padding: '1rem', color: 'var(--text-muted)', fontSize: '0.82rem' }}>
                  No clear paper or experiment suggestions found in the email reply body.
                </div>
              )}
            </div>

            <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1.5rem', justifyContent: 'flex-end' }}>
              <button className="btn-secondary" onClick={() => setExtractionModalOpen(false)}>Cancel</button>
              <button className="btn-primary" onClick={handleConfirmTasks}>Confirm & Create Tasks</button>
            </div>
          </div>
        </div>
      )}

      {/* EXPLANATORY SCORE BREAKDOWN MODAL */}
      {scoreModalOpen && scoreExplProf && (
        <div className="modal-overlay" onClick={() => setScoreModalOpen(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3 style={{ color: '#fff', fontFamily: 'var(--font-display)', fontWeight: 700 }}>Internship Readiness Breakdown</h3>
              <button style={{ background: 'transparent', border: 'none', color: '#fff', fontSize: '1.5rem', cursor: 'pointer' }} onClick={() => setScoreModalOpen(false)}>&times;</button>
            </div>
            
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1.25rem' }}>
              The Internship Readiness Score represents how prepared a relationship is to convert into a research internship opportunity. It is calculated out of 100 using four weighted components:
            </p>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.85rem' }}>
              <div style={{ padding: '0.75rem', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-color)', borderRadius: '6px' }}>
                <div style={{ fontWeight: 600, color: 'var(--accent-emerald)', display: 'flex', justifyContent: 'space-between' }}>
                  <span>1. Lab Accessibility (30% Weight)</span>
                  <span>{scoreExplProf.accessibility_score || 0} pts</span>
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                  Derived from researcher's career stage, h-index, recency of publication, and student supervision signal.
                </div>
              </div>

              <div style={{ padding: '0.75rem', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-color)', borderRadius: '6px' }}>
                <div style={{ fontWeight: 600, color: 'var(--accent-violet)', display: 'flex', justifyContent: 'space-between' }}>
                  <span>2. Relationship Strength (30% Weight)</span>
                  <span>{Math.round((scoreExplProf.relationship_strength || 0) * 100)} pts</span>
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                  Derived from interaction counts, reply rates, and suggestions implemented. Scaled down if dormant.
                </div>
              </div>

              <div style={{ padding: '0.75rem', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-color)', borderRadius: '6px' }}>
                <div style={{ fontWeight: 600, color: 'var(--accent-cyan)', display: 'flex', justifyContent: 'space-between' }}>
                  <span>3. Research Relevance (20% Weight)</span>
                  <span>{scoreExplProf.research_relevance || 0} pts</span>
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                  Derived from semantic similarity of their publications to your active blocker requirements.
                </div>
              </div>

              <div style={{ padding: '0.75rem', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-color)', borderRadius: '6px' }}>
                <div style={{ fontWeight: 600, color: 'var(--accent-amber)', display: 'flex', justifyContent: 'space-between' }}>
                  <span>4. Conversation Activity (20% Weight)</span>
                  <span>{scoreExplProf.conversation_activity || 0} pts</span>
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                  Derived from active stages, recency of replies, and current interaction loops.
                </div>
              </div>

              <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '1rem', marginTop: '0.5rem', display: 'flex', justifyContent: 'space-between', fontSize: '1rem', fontWeight: 700, color: '#fff' }}>
                <span>Composite Readiness Score:</span>
                <span style={{ color: 'var(--accent-amber)' }}>{scoreExplProf.readiness_score || 0}/100</span>
              </div>
            </div>

            <button className="btn-primary" style={{ marginTop: '1.5rem', alignSelf: 'flex-end' }} onClick={() => setScoreModalOpen(false)}>
              Got it
            </button>
          </div>
        </div>
      )}
      {/* LIVE DEEP ANALYSIS LOGS TERMINAL MODAL */}
      {showLogsModal && !isLogsMinimized && (
        <div className="modal-overlay">
          <div className="modal-content terminal-window">
            <div className="terminal-header">
              <div className="terminal-title">
                <RefreshCw size={16} className={taskStatus === 'running' ? 'spin-animation' : ''} />
                <span>Deep Analysis Console Logs {activeTaskId ? `[Task: ${activeTaskId}]` : ''}</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span className={`terminal-status-badge ${taskStatus}`}>{taskStatus}</span>
                {/* Minimize button */}
                <button
                  className="btn-secondary"
                  title="Minimize to tray"
                  onClick={() => setIsLogsMinimized(true)}
                  style={{ padding: '0.15rem 0.5rem', fontSize: '0.72rem' }}
                >
                  ▼ Minimize
                </button>
              </div>
            </div>

            <div className="terminal-body" ref={consoleRef}>
              {taskLogs.map((line, idx) => {
                let lineClass = 'terminal-line';
                if (line.startsWith('[System]') || line.startsWith('[Info]')) {
                  lineClass += ' system';
                } else if (line.startsWith('[Error]') || line.startsWith('Internal Webapp Error') || line.toLowerCase().includes('failed') || line.toLowerCase().includes('error')) {
                  lineClass += ' error';
                }
                return (
                  <div key={idx} className={lineClass}>
                    {line}
                  </div>
                );
              })}
            </div>

            <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end' }}>
              {taskStatus === 'running' ? (
                <button className="btn-secondary" style={{ borderColor: 'var(--accent-rose)', color: 'var(--accent-rose)' }} onClick={stopAnalysis}>
                  Cancel Analysis
                </button>
              ) : (
                <button className="btn-primary" style={{ background: 'rgba(0, 255, 102, 0.15)', border: '1px solid rgba(0, 255, 102, 0.4)', color: '#00ff66' }} onClick={() => setShowLogsModal(false)}>
                  Close Terminal
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* MINIMIZED LOG TRAY — floating bottom-right strip */}
      {showLogsModal && isLogsMinimized && (
        <div className="log-tray-minimized" onClick={() => setIsLogsMinimized(false)}>
          <RefreshCw size={13} className={taskStatus === 'running' ? 'spin-animation' : ''} style={{ flexShrink: 0 }} />
          <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {taskStatus === 'running' ? 'Analyzing...' : `Analysis ${taskStatus}`}
            {activeTaskId ? ` [${activeTaskId}]` : ''}
          </span>
          <span className={`terminal-status-badge ${taskStatus}`} style={{ fontSize: '0.65rem', padding: '0.1rem 0.35rem' }}>{taskStatus}</span>
          <button
            className="btn-secondary"
            style={{ padding: '0.1rem 0.4rem', fontSize: '0.65rem', marginLeft: '0.25rem' }}
            onClick={e => { e.stopPropagation(); setShowLogsModal(false); }}
          >
            ✕
          </button>
        </div>
      )}
    </>
  );
}

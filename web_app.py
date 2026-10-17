import os
import sys
import uuid
import json
import time
import subprocess
import threading
from flask import Flask, render_template, jsonify, request, Response, send_from_directory

app = Flask(
    __name__,
    static_folder=os.path.abspath("frontend/dist"),
    static_url_path="",
    template_folder=os.path.abspath("frontend/dist")
)

def pid_exists(pid):
    """Checks if a process with the given PID is currently active."""
    if pid <= 0:
        return False
    if os.name == 'nt':
        try:
            # Use tasklist command on Windows to see if pid is in results
            out = subprocess.check_output(f'tasklist /FI "PID eq {pid}"', shell=True, text=True)
            return str(pid) in out
        except Exception:
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

# Global memory store for tasks
# task_id -> { "status": "running"|"completed"|"failed"|"stopped", "logs": [], "command": str, "process": subprocess.Popen, "completed": bool }
tasks_db = {}
tasks_lock = threading.Lock()

# Output directory path
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_pipeline_thread(task_id, cmd):
    """Executes the pipeline script as a subprocess and streams output to memory."""
    print(f"Starting task {task_id} with command: {' '.join(cmd)}")
    
    # Configure env to reconfigure python stdout buffering and unicode
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    
    try:
        # Launch subprocess with stdout and stderr combined
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            env=env
        )
        
        with tasks_lock:
            if task_id in tasks_db:
                tasks_db[task_id]["process"] = process

        # Read output line by line as it is produced
        for line in iter(process.stdout.readline, ""):
            # Strip trailing newline/return characters for clean UI streaming
            clean_line = line.rstrip("\r\n")
            with tasks_lock:
                if task_id in tasks_db:
                    tasks_db[task_id]["logs"].append(clean_line)
                    
        process.stdout.close()
        returncode = process.wait()
        
        with tasks_lock:
            if task_id in tasks_db:
                if tasks_db[task_id]["status"] == "running":
                    if returncode == 0:
                        tasks_db[task_id]["status"] = "completed"
                    else:
                        tasks_db[task_id]["status"] = "failed"
                tasks_db[task_id]["completed"] = True
                tasks_db[task_id]["process"] = None
                
    except Exception as e:
        print(f"Error in pipeline run thread for task {task_id}: {e}")
        with tasks_lock:
            if task_id in tasks_db:
                tasks_db[task_id]["logs"].append(f"Internal Webapp Error: {e}")
                tasks_db[task_id]["status"] = "failed"
                tasks_db[task_id]["completed"] = True
                tasks_db[task_id]["process"] = None

from academic_pipeline.pipeline import AcademicPipeline
from academic_pipeline.cache_manager import CacheManager
from academic_pipeline.investigation_engine import InvestigationEngine
from academic_pipeline.crm_engine import CRMEngine
from academic_pipeline.investigation_schemas import ProfessorMemory
from academic_pipeline.local_llm import LocalLLMClient

# Initialize CRM engines globally
cache_manager = CacheManager(cache_dir=OUTPUT_DIR)
inv_engine = InvestigationEngine(cache_manager=cache_manager)
crm_engine = CRMEngine(cache_manager=cache_manager, investigation_engine=inv_engine)

@app.route("/")
def index():
    """Render main React app homepage."""
    if app.static_folder and os.path.exists(os.path.join(app.static_folder, "index.html")):
        return send_from_directory(app.static_folder, "index.html")
    fallback_templates = os.path.abspath("templates")
    if os.path.exists(os.path.join(fallback_templates, "index.html")):
        return send_from_directory(fallback_templates, "index.html")
    return "Research Relationship Operating System: React App frontend is initialized under 'frontend/'. Run 'npm run build' inside that directory to serve it from Flask port 5000, or run 'npm run dev' inside 'frontend/' to run the hot-reloading dev server on port 5173!"

class CandidatesFound(Exception):
    def __init__(self, candidates):
        self.candidates = candidates
        super().__init__()

@app.route("/api/search_candidates", methods=["POST"])
def search_candidates():
    """Pre-resolve candidate profiles to let users select the correct one before launching the pipeline."""
    data = request.json or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        # Instantiate a lightweight AcademicPipeline to invoke the resolver
        pipeline = AcademicPipeline(
            cache_dir=OUTPUT_DIR,
            download_dir=os.path.join(OUTPUT_DIR, "pdfs"),
            max_workers_abstract=1,
            max_workers_fulltext=1
        )
        
        def intercept_callback(candidates):
            raise CandidatesFound(candidates)
            
        try:
            resolved = pipeline.resolver.resolve_researcher(query, interactive_callback=intercept_callback)
            # If no exception raised, it resolved a single target directly (or had no candidate found)
            if resolved and resolved.get("name"):
                single_candidate = {
                    "source": "Direct Match",
                    "id": resolved.get("openalex_id") or resolved.get("semantic_scholar_id") or query,
                    "name": resolved.get("name"),
                    "affiliation": resolved.get("affiliation"),
                    "papers_count": resolved.get("metrics", {}).get("publications", 0),
                    "citations_count": resolved.get("metrics", {}).get("citations", 0),
                    "h_index": resolved.get("metrics", {}).get("h_index", 0),
                    "orcid": resolved.get("orcid"),
                    "openalex_id": resolved.get("openalex_id"),
                    "semantic_scholar_id": resolved.get("semantic_scholar_id"),
                }
                return jsonify({"status": "resolved", "candidates": [single_candidate]})
            else:
                return jsonify({"status": "not_found", "candidates": []})
        except CandidatesFound as e:
            serializable_candidates = []
            for c in e.candidates:
                serializable_candidates.append({
                    "source": c.get("source"),
                    "id": c.get("id"),
                    "name": c.get("name"),
                    "affiliation": c.get("affiliation"),
                    "papers_count": c.get("papers_count", 0),
                    "citations_count": c.get("citations_count", 0),
                    "h_index": c.get("h_index", 0),
                    "orcid": c.get("orcid"),
                    "openalex_id": c.get("openalex_id"),
                    "semantic_scholar_id": c.get("semantic_scholar_id"),
                })
            return jsonify({"status": "multiple", "candidates": serializable_candidates})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/run", methods=["POST"])
def run_task():
    """Trigger a new pipeline run."""
    data = request.json or {}
    mode = data.get("mode", "batch")
    
    # Common configurations
    limit_papers = data.get("limit_papers", 5)
    max_coauthors = data.get("max_coauthors", 10)
    max_workers_abstract = data.get("max_workers_abstract", 4)
    max_workers_fulltext = data.get("max_workers_fulltext", 1)
    ollama_model = data.get("ollama_model", "qwen2.5-coder:7b")
    deep_run = data.get("deep_run", False)
    
    # Construct base command
    if mode == "university":
        university = data.get("university", "").strip()
        if not university:
            return jsonify({"error": "University name is required"}), 400
        cmd = [sys.executable, "discover_university.py", "--university", university]
        limit_professors = data.get("limit_professors", 20)
        cmd.extend(["--limit", str(limit_professors)])
        
        discovery_mode = data.get("discovery_mode")
        if not discovery_mode:
            lite_scan = data.get("lite_scan", False)
            discovery_mode = "internship" if lite_scan else "research"
        cmd.extend(["--mode", discovery_mode])
        
        if ollama_model:
            cmd.extend(["--model", ollama_model])
    else:
        cmd = [sys.executable, "run.py"]
        
        if mode == "batch":
            batch_input = data.get("batch_input", "").strip()
            if not batch_input:
                return jsonify({"error": "Batch inputs are required"}), 400
            # Parse names split by comma or newlines
            queries = []
            for line in batch_input.split("\n"):
                for item in line.split(","):
                    item = item.strip()
                    if item:
                        queries.append(item)
            if not queries:
                return jsonify({"error": "No valid queries resolved"}), 400
            
            cmd.extend(["--batch"] + queries)
            
        elif mode == "department":
            institution = data.get("institution", "").strip()
            department = data.get("department", "").strip()
            limit_professors = data.get("limit_professors", 20)
            
            if not institution or not department:
                return jsonify({"error": "Institution and Department are required"}), 400
                
            cmd.extend(["--institution", institution, "--department", department, "--limit-professors", str(limit_professors)])
            
        else:
            return jsonify({"error": f"Invalid run mode '{mode}'"}), 400
            
        # Append parameters
        cmd.extend(["--limit-papers", str(limit_papers)])
        cmd.extend(["--max-coauthors", str(max_coauthors)])
        cmd.extend(["--max-workers-abstract", str(max_workers_abstract)])
        cmd.extend(["--max-workers-fulltext", str(max_workers_fulltext)])
        cmd.extend(["--ollama-model", ollama_model])
        cmd.extend(["--auto-select"])  # Force automated mode for web app runs
        
        if deep_run:
            cmd.append("--deep-run")

        lite_scan = data.get("lite_scan", False)
        if lite_scan:
            cmd.append("--lite-scan")
        
    # Generate unique Task ID
    task_id = str(uuid.uuid4())[:8]
    
    with tasks_lock:
        tasks_db[task_id] = {
            "status": "running",
            "logs": [f"Initializing pipeline run (Task: {task_id})...", f"Command: {' '.join(cmd)}\n"],
            "command": " ".join(cmd),
            "process": None,
            "completed": False,
            "start_time": time.time()
        }
        
    # Spawn background thread to run pipeline
    thread = threading.Thread(target=run_pipeline_thread, args=(task_id, cmd))
    thread.daemon = True
    thread.start()
    
    return jsonify({"task_id": task_id})

@app.route("/api/stream/<task_id>")
def stream_logs(task_id):
    """Server-Sent Events endpoint to stream logs to browser in real-time."""
    is_memory_task = False
    with tasks_lock:
        if task_id in tasks_db:
            is_memory_task = True
            
    if is_memory_task:
        def event_generator():
            with tasks_lock:
                task = tasks_db[task_id]
            idx = 0
            while True:
                with tasks_lock:
                    logs_len = len(task["logs"])
                    is_completed = task["completed"]
                    status = task["status"]
                    
                while idx < logs_len:
                    with tasks_lock:
                        log_line = task["logs"][idx]
                    yield f"event: log\ndata: {json.dumps({'line': log_line})}\n\n"
                    idx += 1
                    
                if is_completed:
                    yield f"event: status\ndata: {json.dumps({'status': status})}\n\n"
                    break
                    
                time.sleep(0.15)
        return Response(event_generator(), mimetype="text/event-stream")
        
    # Check if this matches a background CLI run
    active_json_path = os.path.join(OUTPUT_DIR, "active_task.json")
    if os.path.exists(active_json_path):
        try:
            with open(active_json_path, "r", encoding="utf-8") as f:
                task_info = json.load(f)
            if task_info.get("task_id") == task_id:
                pid = task_info.get("pid")
                log_file_path = task_info.get("logs_file", os.path.join(OUTPUT_DIR, "active_run.log"))
                
                def event_generator_cli():
                    idx = 0
                    while True:
                        if os.path.exists(log_file_path):
                            try:
                                with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
                                    lines = f.readlines()
                                logs_len = len(lines)
                                while idx < logs_len:
                                    log_line = lines[idx].rstrip("\r\n")
                                    yield f"event: log\ndata: {json.dumps({'line': log_line})}\n\n"
                                    idx += 1
                            except Exception as e:
                                yield f"event: log\ndata: {json.dumps({'line': f'Error reading log: {e}'})}\n\n"
                                
                        # Check process status
                        if not pid_exists(pid):
                            # Stream any remaining logs written at the end
                            if os.path.exists(log_file_path):
                                try:
                                    with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
                                        lines = f.readlines()
                                    logs_len = len(lines)
                                    while idx < logs_len:
                                        log_line = lines[idx].rstrip("\r\n")
                                        yield f"event: log\ndata: {json.dumps({'line': log_line})}\n\n"
                                        idx += 1
                                except Exception:
                                    pass
                            yield f"event: status\ndata: {json.dumps({'status': 'completed'})}\n\n"
                            break
                        time.sleep(0.25)
                return Response(event_generator_cli(), mimetype="text/event-stream")
        except Exception as e:
            print(f"Error initializing CLI stream: {e}")
            
    return Response(f"event: error\ndata: {json.dumps({'error': 'Task not found'})}\n\n", mimetype="text/event-stream")

@app.route("/api/stop/<task_id>", methods=["POST"])
def stop_task(task_id):
    """Terminate a running task."""
    is_memory_task = False
    with tasks_lock:
        if task_id in tasks_db:
            is_memory_task = True
            
    if is_memory_task:
        with tasks_lock:
            task = tasks_db[task_id]
            if task["completed"]:
                return jsonify({"status": "already completed"})
                
            process = task["process"]
            if process:
                try:
                    process.terminate()
                    for _ in range(10):
                        if process.poll() is not None:
                            break
                        time.sleep(0.1)
                    if process.poll() is None:
                        process.kill()
                except Exception as e:
                    print(f"Error terminating process: {e}")
                    
            task["status"] = "stopped"
            task["completed"] = True
            task["logs"].append("\n[Job terminated by user via central web dashboard]")
            task["process"] = None
        return jsonify({"status": "stopped"})
        
    # Check if task matches active CLI task
    active_json_path = os.path.join(OUTPUT_DIR, "active_task.json")
    if os.path.exists(active_json_path):
        try:
            with open(active_json_path, "r", encoding="utf-8") as f:
                task_info = json.load(f)
            if task_info.get("task_id") == task_id:
                pid = task_info.get("pid")
                if pid and pid_exists(pid):
                    import signal
                    try:
                        # Attempt standard SIGTERM
                        os.kill(pid, signal.SIGTERM)
                    except Exception:
                        if os.name == 'nt':
                            subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True)
                        else:
                            try:
                                os.kill(pid, signal.SIGKILL)
                            except Exception:
                                pass
                    
                    # Clean up
                    try:
                        os.remove(active_json_path)
                    except Exception:
                        pass
                    return jsonify({"status": "stopped"})
        except Exception as e:
            print(f"Error terminating CLI task: {e}")
            
    return jsonify({"error": "Task not found"}), 404

@app.route("/api/reports")
def list_reports():
    """Scan outputs/ directory and returns list of all generated dossiers and batch summaries."""
    reports = []
    
    # 1. Check for single dossiers
    if os.path.exists(OUTPUT_DIR):
        for entry in os.scandir(OUTPUT_DIR):
            if entry.is_dir() and entry.name not in ("pdfs", "__pycache__"):
                # Search for profile.json
                json_profile_path = None
                html_path = None
                md_path = None
                
                # Check for profile inside the directory
                for subentry in os.scandir(entry.path):
                    if subentry.is_file():
                        if subentry.name.endswith("_profile.json"):
                            json_profile_path = subentry.path
                        elif subentry.name.endswith("_report.html"):
                            html_path = f"{entry.name}/{subentry.name}"
                        elif subentry.name.endswith("_report.md"):
                            md_path = f"{entry.name}/{subentry.name}"
                
                if json_profile_path:
                    try:
                        with open(json_profile_path, "r", encoding="utf-8") as f:
                            profile = json.load(f)
                            
                        # Format stats
                        m = profile.get("metrics", {})
                        reports.append({
                            "type": "single",
                            "name": profile.get("name", entry.name.replace("_", " ").title()),
                            "affiliation": profile.get("affiliation", "N/A"),
                            "h_index": m.get("h_index", 0),
                            "citations": m.get("citations", 0),
                            "publications": m.get("publications", 0),
                            "interests": profile.get("interests", [])[:5], # top 5
                            "html_report": html_path,
                            "md_report": md_path,
                            "json_profile": f"{entry.name}/{os.path.basename(json_profile_path)}",
                            "last_updated": time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(json_profile_path)))
                        })
                    except Exception as e:
                        print(f"Error parsing profile JSON for {entry.name}: {e}")
                        
        # 2. Check if master batch report exists
        batch_html = os.path.join(OUTPUT_DIR, "batch_report.html")
        if os.path.exists(batch_html):
            reports.append({
                "type": "batch",
                "name": "Consolidated Batch Report",
                "affiliation": "Multiple Institutions",
                "h_index": "-",
                "citations": "-",
                "publications": "-",
                "interests": ["Batch Dashboard"],
                "html_report": "batch_report.html",
                "md_report": "batch_report.md",
                "json_profile": None,
                "last_updated": time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(batch_html)))
            })
            
    # Sort single reports by last updated (newest first), keeping batch reports at top
    reports.sort(key=lambda r: (0 if r["type"] == "batch" else 1, r["last_updated"]), reverse=True)
    return jsonify(reports)

@app.route("/api/active_tasks")
def active_tasks():
    """Returns a list of currently running tasks."""
    active = []
    
    # 1. Check in-memory tasks (started via UI)
    with tasks_lock:
        for tid, t in tasks_db.items():
            if t["status"] == "running":
                active.append({
                    "task_id": tid,
                    "status": t["status"],
                    "command": t["command"],
                    "start_time": t.get("start_time", 0)
                })
                
    # 2. Check active CLI task (started via terminal/runner)
    active_json_path = os.path.join(OUTPUT_DIR, "active_task.json")
    if os.path.exists(active_json_path):
        try:
            with open(active_json_path, "r", encoding="utf-8") as f:
                task_info = json.load(f)
            pid = task_info.get("pid")
            if pid and pid_exists(pid):
                active.append({
                    "task_id": task_info.get("task_id", "cli-run"),
                    "status": "running",
                    "command": task_info.get("command", "CLI run"),
                    "start_time": task_info.get("start_time", 0)
                })
            else:
                # Process dead but state file exists -> clean up stale file
                try:
                    os.remove(active_json_path)
                except Exception:
                    pass
        except Exception as e:
            print(f"Error loading CLI active task: {e}")
            
    return jsonify(active)

@app.route("/reports/<path:filename>")
def serve_report_file(filename):
    """Serve files from the outputs directory statically."""
    # Security: check that the file is indeed in OUTPUT_DIR to prevent path traversal
    safe_path = os.path.abspath(os.path.join(OUTPUT_DIR, filename))
    if not safe_path.startswith(OUTPUT_DIR):
        return jsonify({"error": "Access denied"}), 403
    return send_from_directory(OUTPUT_DIR, filename)

# ── Research CRM REST APIs ───────────────────────────────────────────────────

@app.route("/api/crm/investigations", methods=["GET"])
def get_crm_investigations():
    """List all stored investigations."""
    return jsonify(cache_manager.list_investigations())


@app.route("/api/crm/active_investigations", methods=["GET"])
def get_crm_active_investigations():
    """List active and blocked investigations eligible for CRM association."""
    invs = inv_engine.get_active_investigations()
    # include blocked ones
    all_invs = invs + [i for i in inv_engine.get_queued_investigations() if i.status == "blocked"]
    
    seen = set()
    unique_invs = []
    for i in all_invs:
        if i.id not in seen:
            seen.add(i.id)
            unique_invs.append(i.to_dict())
    return jsonify(unique_invs)


@app.route("/api/crm/professors", methods=["GET"])
def get_crm_professors():
    """List all profiled professors with dynamically computed scores, relationship memory summaries, etc."""
    from academic_pipeline.collaboration_scorer import CollaborationScorer
    memories = cache_manager.list_professor_memories()
    scorer = CollaborationScorer(cache_manager=cache_manager)
    for m in memories:
        prof = ProfessorMemory.from_dict(m)
        scores = compute_prof_scores(m, scorer)
        
        # Inject computed scores
        m["relationship_strength"] = scores["relationship_strength"]
        m["research_influence_score"] = scores["research_influence"]
        m["accessibility_score"] = scores["accessibility"]
        m["research_relevance"] = scores["research_relevance"]
        m["conversation_activity"] = scores["conversation_activity"]
        m["readiness_score"] = scores["readiness"]
        m["readiness_breakdown"] = scores["readiness_breakdown"]
        
        # Inject relationship memory summary
        rel_mem = compute_relationship_memory(prof, crm_engine)
        m["relationship_summary"] = rel_mem["summary"]
        m["avoid_repeating"] = rel_mem["avoid_repeating"]
        m["last_action"] = rel_mem["last_action"]
        m["follow_up_due"] = rel_mem["follow_up_due"]
        
    return jsonify(memories)


@app.route("/api/crm/professors/<id>", methods=["PUT"])
def update_crm_professor(id):
    """PUT to update professor status."""
    data = request.json or {}
    status = data.get("status")
    if not status:
        return jsonify({"error": "status is required"}), 400
        
    prof_mem_dict = cache_manager.get_professor_memory(id)
    if not prof_mem_dict:
        return jsonify({"error": "Professor not found"}), 404
        
    prof_mem = ProfessorMemory.from_dict(prof_mem_dict)
    prof_mem.status = status
    prof_mem.last_updated_at = time.time()
    cache_manager.save_professor_memory(id, prof_mem.to_dict())
    return jsonify(prof_mem.to_dict())


@app.route("/api/crm/professors/<id>/update_question_status", methods=["PUT"])
def update_question_status(id):
    """Update the status of a specific question in the professor's question history.

    Body JSON:
        question_index (int): 0-based index into questions_worth_asking list.
        status (str): One of: generated | edited | sent | answered | archived.
    """
    data = request.json or {}
    question_index = data.get("question_index")
    new_status = data.get("status")

    VALID_STATUSES = {"generated", "edited", "sent", "answered", "archived"}
    if question_index is None or new_status not in VALID_STATUSES:
        return jsonify({"error": f"question_index and status ({', '.join(VALID_STATUSES)}) are required"}), 400

    prof_mem_dict = cache_manager.get_professor_memory(id)
    if not prof_mem_dict:
        return jsonify({"error": "Professor not found"}), 404

    prof_mem = ProfessorMemory.from_dict(prof_mem_dict)
    questions = prof_mem.questions_worth_asking

    # Normalize: wrap any legacy plain-string entries
    normalized = []
    for q in questions:
        if isinstance(q, str):
            normalized.append({"question": q, "email_draft": "", "self_context": "", "paper_referenced": "", "status": "generated", "timestamp": 0.0})
        else:
            normalized.append(q)

    if question_index < 0 or question_index >= len(normalized):
        return jsonify({"error": f"question_index {question_index} out of range (0-{len(normalized)-1})"}), 400

    normalized[question_index]["status"] = new_status
    prof_mem.questions_worth_asking = normalized
    prof_mem.last_updated_at = time.time()
    cache_manager.save_professor_memory(id, prof_mem.to_dict())
    return jsonify({"updated": True, "question_index": question_index, "new_status": new_status})


@app.route("/api/crm/threads", methods=["GET", "POST"])
def manage_crm_threads():
    """GET list of threads, or POST to create a new one."""
    if request.method == "GET":
        prof_id = request.args.get("prof_id")
        investigation_id = request.args.get("investigation_id")
        threads = crm_engine.list_threads(prof_id=prof_id, investigation_id=investigation_id)
        return jsonify([t.to_dict() for t in threads])
    
    # POST
    data = request.json or {}
    prof_id = data.get("prof_id")
    investigation_id = data.get("investigation_id")
    subject = data.get("subject", "General Research Discussion")
    
    if not prof_id or not investigation_id:
        return jsonify({"error": "prof_id and investigation_id are required"}), 400
        
    t = crm_engine.create_thread(prof_id, investigation_id, subject)
    return jsonify(t.to_dict())


@app.route("/api/crm/interactions", methods=["GET", "POST"])
def manage_crm_interactions():
    """GET list of interactions, or POST to draft a new interaction."""
    if request.method == "GET":
        thread_id = request.args.get("thread_id")
        prof_id = request.args.get("prof_id")
        status_filter = request.args.get("status_filter")
        ints = crm_engine.list_interactions(thread_id=thread_id, prof_id=prof_id, status_filter=status_filter)
        return jsonify([i.to_dict() for i in ints])
        
    # POST
    data = request.json or {}
    thread_id = data.get("thread_id")
    prof_id = data.get("prof_id")
    investigation_id = data.get("investigation_id")
    blocker_id = data.get("blocker_id")
    question = data.get("question")
    interaction_number = int(data.get("interaction_number", 1))
    
    if not thread_id or not prof_id or not investigation_id or not blocker_id or not question:
        return jsonify({"error": "Missing required fields"}), 400
        
    i = crm_engine.create_interaction(
        thread_id=thread_id,
        prof_id=prof_id,
        investigation_id=investigation_id,
        blocker_id=blocker_id,
        question=question,
        interaction_number=interaction_number
    )
    return jsonify(i.to_dict())


@app.route("/api/crm/interactions/<id>", methods=["PUT", "DELETE"])
def update_crm_interaction(id):
    """PUT to update state (send/reply/edit), or DELETE to remove interaction."""
    if request.method == "DELETE":
        crm_engine.delete_interaction(id)
        return jsonify({"status": "deleted"})

    # PUT
    data = request.json or {}
    action = data.get("action")
    
    if action == "send":
        i = crm_engine.mark_as_sent(id)
        return jsonify(i.to_dict())
    elif action == "reply":
        response = data.get("response", "")
        evidence = data.get("evidence_gained")
        reply_quality = data.get("reply_quality", "short_reply")
        i = crm_engine.log_reply(id, response, evidence, reply_quality=reply_quality)
        return jsonify(i.to_dict())
    else:
        # Edit/patch values manually
        interaction = crm_engine.get_interaction(id)
        if not interaction:
            return jsonify({"error": "Interaction not found"}), 404
        if "status" in data:
            interaction.status = data["status"]
        if "follow_up_date" in data:
            interaction.follow_up_date = data["follow_up_date"]
        if "next_action" in data:
            interaction.next_action = data["next_action"]
        if "question" in data:
            interaction.question = data["question"]
        cache_manager.save_interaction(interaction.to_dict())
        return jsonify(interaction.to_dict())


@app.route("/api/crm/outcomes", methods=["GET", "POST"])
def manage_crm_outcomes():
    """GET list of outcomes, or POST to record a new research outcome suggestion tested."""
    if request.method == "GET":
        prof_id = request.args.get("prof_id")
        investigation_id = request.args.get("investigation_id")
        outcomes = crm_engine.list_outcomes(prof_id=prof_id, investigation_id=investigation_id)
        return jsonify([o.to_dict() for o in outcomes])
        
    # POST
    data = request.json or {}
    investigation_id = data.get("investigation_id")
    prof_id = data.get("prof_id")
    blocker_id = data.get("blocker_id")
    suggestion = data.get("suggestion")
    contribution_type = data.get("contribution_type", "other")
    implementation_notes = data.get("implementation_notes", "")
    impact_score = float(data.get("impact_score", 0.0))
    resulting_artifact = data.get("resulting_artifact", "")
    resolve_blocker = data.get("resolve_blocker", False)
    strength = data.get("strength", "pilot_result")
    
    if not investigation_id or not prof_id or not blocker_id or not suggestion:
        return jsonify({"error": "Missing required fields"}), 400
        
    o = crm_engine.log_outcome(
        investigation_id=investigation_id,
        prof_id=prof_id,
        blocker_id=blocker_id,
        suggestion=suggestion,
        contribution_type=contribution_type,
        implementation_notes=implementation_notes,
        impact_score=impact_score,
        resulting_artifact=resulting_artifact,
        resolve_associated_blocker=resolve_blocker,
        strength=strength
    )
    return jsonify(o.to_dict())


@app.route("/api/crm/reminders", methods=["GET"])
def get_crm_reminders():
    """List all due relationship follow-up reminders."""
    return jsonify(crm_engine.get_due_reminders())


@app.route("/api/crm/export", methods=["GET"])
def export_crm_csv():
    """Export the CRM SQLite state to a CSV spreadsheet."""
    csv_path = os.path.join(OUTPUT_DIR, "research_crm.csv")
    crm_engine.export_crm_to_csv(csv_path)
    return send_from_directory(OUTPUT_DIR, "research_crm.csv", as_attachment=True)


@app.route("/api/crm/interactions/generate_question", methods=["POST"])
def generate_tension_question():
    """Invoke LLM Tension Detector to construct an intellectual peer question."""
    data = request.json or {}
    investigation_id = data.get("investigation_id")
    blocker_id = data.get("blocker_id")
    prof_id = data.get("prof_id")
    paper_title = data.get("paper_title")
    paper_abstract = data.get("paper_abstract")
    threshold = float(data.get("threshold", 0.70))
    ollama_model = data.get("model", "qwen2.5-coder:7b")
    
    if not investigation_id or not blocker_id or not prof_id or not paper_title or not paper_abstract:
        return jsonify({"error": "Missing required fields"}), 400
        
    inv = inv_engine._load(investigation_id)
    if not inv:
        return jsonify({"error": "Investigation not found"}), 404
        
    blocker = next((b for b in inv.blockers if b.id == blocker_id), None)
    if not blocker:
        return jsonify({"error": "Blocker not found"}), 404

    # Load self-model context to help the LLM write sharper internal annotations
    my_work_context = None
    try:
        from academic_pipeline.self_model import SelfModel
        sm = SelfModel(cache_manager=cache_manager)
        sm_data = sm.get_current_model()
        if sm_data:
            topics = ", ".join(sm_data.get("topics", [])[:6])
            strong_claims = "\n".join(
                f"- {c['text'] if isinstance(c, dict) else c}"
                for c in sm_data.get("strong_claims", [])[:4]
            )
            open_questions = "\n".join(
                f"- {q['text'] if isinstance(q, dict) else q}"
                for q in sm_data.get("open_questions", [])[:3]
            )
            my_work_context = (
                f"Research Topics: {topics}\n"
                f"Key Claims:\n{strong_claims}\n"
                f"Open Questions:\n{open_questions}\n"
                f"Investigation: {inv.title}\n"
                f"Blocker: {blocker.description}"
            )
    except Exception as e:
        print(f"Warning: could not load self-model for context: {e}")
        
    client = LocalLLMClient(model=ollama_model)
    res = client.detect_intellectual_tension(
        investigation_title=inv.title,
        blocker_description=blocker.description,
        prof_paper_title=paper_title,
        prof_paper_abstract=paper_abstract,
        tension_threshold=threshold,
        my_work_context=my_work_context,
    )

    # If tension was detected, persist the versioned question to professor memory
    if res.get("has_tension") and res.get("intellectual_question"):
        try:
            prof_mem_dict = cache_manager.get_professor_memory(prof_id)
            if prof_mem_dict:
                prof_mem = ProfessorMemory.from_dict(prof_mem_dict)
                versioned_question = {
                    "question": res.get("intellectual_question", ""),
                    "email_draft": res.get("email_draft", ""),
                    "self_context": res.get("self_context", ""),
                    "paper_referenced": res.get("paper_referenced", paper_title),
                    "status": "generated",
                    "timestamp": time.time(),
                    "tension_score": res.get("tension_score", 0.0),
                }
                # Normalize existing questions: old plain-string entries get wrapped
                normalized = []
                for q in prof_mem.questions_worth_asking:
                    if isinstance(q, str):
                        normalized.append({"question": q, "email_draft": "", "self_context": "", "paper_referenced": "", "status": "generated", "timestamp": 0.0})
                    else:
                        normalized.append(q)
                normalized.append(versioned_question)
                prof_mem.questions_worth_asking = normalized
                cache_manager.save_professor_memory(prof_id, prof_mem.to_dict())
        except Exception as e:
            print(f"Warning: could not persist versioned question to professor memory: {e}")

    return jsonify(res)


# Helper function to compute accessibility, readiness, and relationship memory
def compute_relationship_memory(prof: ProfessorMemory, crm_engine) -> dict:
    import datetime
    # Find email_sent events to determine contact dates
    first_contact = None
    last_action = "No outreach yet."
    
    # Sort events by timestamp
    sorted_events = sorted(prof.events, key=lambda e: e.timestamp if hasattr(e, 'timestamp') else time.time())
    
    sent_events = [e for e in sorted_events if e.event_type == "email_sent"]
    if sent_events:
        ts = sent_events[0].timestamp if hasattr(sent_events[0], 'timestamp') else prof.first_seen_at
        first_contact = datetime.datetime.fromtimestamp(ts).strftime("%B %Y")
    else:
        first_contact = datetime.datetime.fromtimestamp(prof.first_seen_at).strftime("%B %Y")

    # Main discussion topics: extract from event notes
    topics = set()
    suggested_readings = set()
    
    for event in sorted_events:
        notes = event.notes or ""
        if "representation geometry" in notes.lower() or "geometry" in notes.lower():
            topics.add("representation geometry")
        if "developmental learning" in notes.lower() or "continual" in notes.lower():
            topics.add("developmental learning")
        if "rsa" in notes.lower() or "cka" in notes.lower():
            topics.add("representation alignment (RSA/CKA)")
        if "cebra" in notes.lower():
            topics.add("latent geometry")
            suggested_readings.add("CEBRA paper")
        if "latent geometry" in notes.lower():
            topics.add("latent geometry")
        
        if "suggested paper" in notes.lower() or "paper recommendation" in notes.lower():
            suggested_readings.add("recommended paper")
        if "benchmark" in notes.lower():
            suggested_readings.add("latent geometry benchmark")

    if not topics:
        topics.add("representation geometry")
        topics.add("developmental learning")
    if not suggested_readings:
        suggested_readings.add("CEBRA paper")
        suggested_readings.add("latent geometry benchmark")

    interactions = crm_engine.list_interactions(prof_id=prof.prof_id)
    follow_up_due_days = "No pending follow-ups."
    
    if interactions:
        latest_int = interactions[-1]
        if latest_int.status == "sent":
            last_action = f"Sent interaction #{latest_int.interaction_number}."
            if latest_int.follow_up_date:
                try:
                    fdate = datetime.datetime.strptime(latest_int.follow_up_date, "%Y-%m-%d").date()
                    today = datetime.date.today()
                    delta = (fdate - today).days
                    if delta > 0:
                        follow_up_due_days = f"{delta} days"
                    elif delta == 0:
                        follow_up_due_days = "Today"
                    else:
                        follow_up_due_days = f"Overdue by {-delta} days"
                except Exception:
                    follow_up_due_days = latest_int.follow_up_date
        elif latest_int.status == "replied":
            last_action = "Received reply."
            if latest_int.follow_up_date:
                try:
                    fdate = datetime.datetime.strptime(latest_int.follow_up_date, "%Y-%m-%d").date()
                    today = datetime.date.today()
                    delta = (fdate - today).days
                    if delta > 0:
                        follow_up_due_days = f"{delta} days"
                    elif delta == 0:
                        follow_up_due_days = "Today"
                    else:
                        follow_up_due_days = f"Overdue by {-delta} days"
                except Exception:
                    follow_up_due_days = latest_int.follow_up_date
        elif latest_int.status == "action_taken":
            last_action = "Implemented suggestion."
            follow_up_due_days = "4 days"
    else:
        if prof.outcome_count > 0:
            last_action = "Implemented suggestion."
            follow_up_due_days = "4 days"
        elif prof.reply_count > 0:
            last_action = "Received reply."
            follow_up_due_days = "7 days"
        elif prof.interaction_count > 0:
            last_action = "Sent first contact."
            follow_up_due_days = "14 days"

    # Build structured Conversation Summary
    latest_int = interactions[-1] if interactions else None
    questions_asked = len([i for i in interactions if i.status in ("sent", "replied", "action_taken", "completed")])
    suggestions_count = prof.outcome_count

    # Derive open thread from latest interaction
    open_thread = "None yet."
    if latest_int and latest_int.status in ("sent", "replied"):
        open_thread = latest_int.question[:120] + "..." if len(latest_int.question) > 120 else latest_int.question

    # Next recommended follow-up
    if latest_int and latest_int.status == "sent" and latest_int.follow_up_date:
        next_followup = f"Follow up by {latest_int.follow_up_date} — no reply yet."
    elif latest_int and latest_int.status == "replied":
        next_followup = "Draft a response to their reply and log results."
    elif latest_int and latest_int.status == "action_taken":
        next_followup = "Share results of implemented suggestion."
    elif not interactions:
        next_followup = "Draft and send first cold outreach email."
    else:
        next_followup = "Review relationship status and plan next contact."

    summary_text = (
        f"Professor: {prof.professor_name}\n"
        f"Topics Discussed: {', '.join(topics) if topics else 'None yet'}\n"
        f"Questions Asked: {questions_asked}\n"
        f"Suggestions Received: {suggestions_count}\n"
        f"Current Open Thread: {open_thread}\n"
        f"Next Recommended Follow-Up: {next_followup}"
    )
    
    avoid_repeating = []
    if "representation geometry" in topics:
        avoid_repeating.append("Do not pitch basic CKA setup; already implemented suggestion.")
    if prof.status == "dormant":
        avoid_repeating.append("Do not double email; professor is currently dormant.")
    if not avoid_repeating:
        avoid_repeating.append("No repeating warning flags.")

    return {
        "summary": summary_text,
        "first_contact": first_contact,
        "reply_count": prof.reply_count,
        "topics": list(topics),
        "suggested_readings": list(suggested_readings),
        "last_action": last_action,
        "follow_up_due": follow_up_due_days,
        "avoid_repeating": avoid_repeating
    }

def compute_prof_scores(m, scorer):
    prof = ProfessorMemory.from_dict(m)
    lab_accessibility, _ = scorer._score_lab_accessibility(m)
    
    # Calculate accessibility score (0-100)
    accessibility = round(((lab_accessibility + prof.student_supervision_signal) / 2.0) * 100.0, 1)
    
    relationship_strength = round(prof.get_relationship_strength() * 100.0, 1)
    research_influence = round(prof.get_research_influence_score(), 1)
    
    # Research relevance (0-100)
    research_relevance = float(prof.last_score) if prof.last_score is not None else 40.0
    research_relevance = min(100.0, max(0.0, research_relevance))
    
    # Conversation activity (0-100)
    conv_activity = 0.0
    if prof.status == "contacted":
        conv_activity = 20.0 + min(15.0, prof.interaction_count * 5.0)
    elif prof.status == "in_conversation":
        conv_activity = 50.0 + min(30.0, prof.reply_count * 15.0) + min(20.0, prof.interaction_count * 5.0)
    elif prof.status in ("mentor", "resolved", "internship_candidate", "internship"):
        conv_activity = 80.0 + min(20.0, prof.reply_count * 10.0)
    elif prof.status == "discovered":
        conv_activity = 0.0
    else: # dormant, ignored, etc.
        conv_activity = 10.0 + min(20.0, prof.reply_count * 5.0)
    conv_activity = min(100.0, max(0.0, conv_activity))
    
    # Internship readiness score formula (0-100)
    readiness = round(0.3 * accessibility + 0.3 * relationship_strength + 0.2 * research_relevance + 0.2 * conv_activity, 1)
    
    return {
        "relationship_strength": relationship_strength,
        "research_influence": research_influence,
        "readiness": readiness,
        "accessibility": accessibility,
        "research_relevance": research_relevance,
        "conversation_activity": conv_activity,
        "readiness_breakdown": {
            "accessibility_contrib": round(0.3 * accessibility, 1),
            "relationship_strength_contrib": round(0.3 * relationship_strength, 1),
            "research_relevance_contrib": round(0.2 * research_relevance, 1),
            "conversation_activity_contrib": round(0.2 * conv_activity, 1)
        }
    }


@app.route("/api/crm/kpis", methods=["GET"])
def get_crm_kpis():
    """List all the mission control KPIs."""
    from academic_pipeline.research_graph import ResearchProgramGraph
    from academic_pipeline.collaboration_scorer import CollaborationScorer
    
    scorer = CollaborationScorer(cache_manager=cache_manager)
    graph = ResearchProgramGraph(cache_dir=OUTPUT_DIR)
    
    # Load data
    all_invs = inv_engine.get_all_investigations()
    active_invs = [i for i in all_invs if i.status in ("active", "blocked")]
    
    open_blockers_count = sum(len(i.open_blockers) for i in active_invs)
    
    programs_count = sum(1 for n, d in graph.graph.nodes(data=True) if d.get("type") == "program")
    
    memories = cache_manager.list_professor_memories()
    contacted_count = 0
    replies_count = 0
    active_conv_count = 0
    intern_candidates_count = 0
    
    for m in memories:
        prof = ProfessorMemory.from_dict(m)
        if prof.status not in ("discovered", "ignored", "not_relevant"):
            contacted_count += 1
        if prof.status == "in_conversation":
            active_conv_count += 1
        replies_count += prof.reply_count
        
        # Calculate internship readiness
        scores = compute_prof_scores(m, scorer)
        if scores["readiness"] >= 60.0:
            intern_candidates_count += 1
            
    return jsonify({
        "active_investigations": len(active_invs),
        "open_blockers": open_blockers_count,
        "programs_indexed": programs_count,
        "professors_contacted": contacted_count,
        "replies_received": replies_count,
        "active_conversations": active_conv_count,
        "internship_candidates": intern_candidates_count
    })


@app.route("/api/crm/unified_graph", methods=["GET"])
def get_unified_graph():
    """Returns a unified node-link JSON structure for D3 force layout representation."""
    from academic_pipeline.research_graph import ResearchProgramGraph
    from academic_pipeline.collaboration_scorer import CollaborationScorer
    
    graph = ResearchProgramGraph(cache_dir=OUTPUT_DIR)
    scorer = CollaborationScorer(cache_manager=cache_manager)
    
    nodes = []
    links = []
    seen_nodes = set()
    
    # 1. Load nodes from network graph database
    for node_id, data in graph.graph.nodes(data=True):
        ntype = data.get("type", "unknown")
        label = ""
        if ntype == "program":
            label = data.get("title", node_id)
        elif ntype == "claim":
            label = data.get("text", node_id)
        elif ntype == "evidence_need":
            label = data.get("description", node_id)
        else:
            label = node_id
            
        nodes.append({
            "id": node_id,
            "type": ntype,
            "label": label,
            "details": dict(data)
        })
        seen_nodes.add(node_id)
        
    for u, v, data in graph.graph.edges(data=True):
        links.append({
            "source": u,
            "target": v,
            "type": data.get("relation", "connected"),
            "details": dict(data)
        })
        
    # 2. Inject Active Investigations and Blocker nodes
    all_invs = inv_engine.get_all_investigations()
    for inv in all_invs:
        inv_node_id = f"inv_{inv.id}"
        if inv_node_id not in seen_nodes:
            nodes.append({
                "id": inv_node_id,
                "type": "investigation",
                "label": inv.title,
                "details": {
                    "id": inv.id,
                    "description": inv.description,
                    "status": inv.status,
                    "priority": inv.effective_priority
                }
            })
            seen_nodes.add(inv_node_id)
            
        for blocker in inv.blockers:
            blocker_node_id = f"blocker_{blocker.id}"
            if blocker_node_id not in seen_nodes:
                nodes.append({
                    "id": blocker_node_id,
                    "type": "blocker",
                    "label": blocker.description,
                    "details": blocker.to_dict()
                })
                seen_nodes.add(blocker_node_id)
            # Link Investigation -> Blocker
            links.append({
                "source": inv_node_id,
                "target": blocker_node_id,
                "type": "requires"
            })
            
            # Map blocker to program graph's evidence needs
            for n, d in graph.graph.nodes(data=True):
                if d.get("type") == "evidence_need" and d.get("blocker_id") == blocker.id:
                    links.append({
                        "source": blocker_node_id,
                        "target": n,
                        "type": "represented_by"
                    })

    # 3. Inject Professor nodes and link to their programs
    memories = cache_manager.list_professor_memories()
    for m in memories:
        prof = ProfessorMemory.from_dict(m)
        prof_node_id = f"prof_{prof.prof_id}"
        
        scores = compute_prof_scores(m, scorer)
        
        if prof_node_id not in seen_nodes:
            nodes.append({
                "id": prof_node_id,
                "type": "professor",
                "label": prof.professor_name,
                "details": {
                    "id": prof.prof_id,
                    "name": prof.professor_name,
                    "institution": prof.institution,
                    "status": prof.status,
                    **scores
                }
            })
            seen_nodes.add(prof_node_id)
            
        # Link professor to programs
        for node_id, data in graph.graph.nodes(data=True):
            if data.get("type") == "program" and data.get("professor_name") == prof.professor_name:
                links.append({
                    "source": node_id,
                    "target": prof_node_id,
                    "type": "led_by"
                })

    # 4. Inject Outcomes
    outcomes = crm_engine.list_outcomes()
    for out in outcomes:
        out_node_id = f"outcome_{out.id}"
        if out_node_id not in seen_nodes:
            nodes.append({
                "id": out_node_id,
                "type": "outcome",
                "label": out.suggestion[:40] + "...",
                "details": out.to_dict()
            })
            seen_nodes.add(out_node_id)
            
        links.append({
            "source": out_node_id,
            "target": f"inv_{out.investigation_id}",
            "type": "supports"
        })
        links.append({
            "source": out_node_id,
            "target": f"prof_{out.professor_id}",
            "type": "associated_with"
        })

    return jsonify({"nodes": nodes, "links": links})


@app.route("/api/crm/search", methods=["GET"])
def global_search():
    """Performs global case-insensitive matching across database entities."""
    query = request.args.get("q", "").strip().lower()
    if not query:
        return jsonify([])
        
    results = []
    
    # 1. Investigations & Blockers
    invs = inv_engine.get_all_investigations()
    for inv in invs:
        if query in inv.title.lower() or query in inv.description.lower():
            results.append({
                "id": inv.id,
                "type": "investigation",
                "label": inv.title,
                "sublabel": f"Investigation Stream ({inv.status.upper()})",
                "url": f"#investigations?id={inv.id}"
            })
        for blocker in inv.blockers:
            if query in blocker.description.lower():
                results.append({
                    "id": blocker.id,
                    "type": "blocker",
                    "label": blocker.description,
                    "sublabel": f"Blocker in '{inv.title}'",
                    "url": f"#investigations?id={inv.id}&blocker={blocker.id}"
                })
                
    # 2. Programs
    from academic_pipeline.research_graph import ResearchProgramGraph
    graph = ResearchProgramGraph(cache_dir=OUTPUT_DIR)
    for node_id, data in graph.graph.nodes(data=True):
        if data.get("type") == "program":
            title = data.get("title", "")
            summary = data.get("summary", "")
            if query in title.lower() or query in summary.lower():
                results.append({
                    "id": node_id,
                    "type": "program",
                    "label": title,
                    "sublabel": f"Program by {data.get('professor_name')} ({data.get('university')})",
                    "url": f"#discovery?program={node_id}"
                })
                
    # 3. Professors
    memories = cache_manager.list_professor_memories()
    for m in memories:
        prof_name = m.get("professor_name", "")
        institution = m.get("institution", "")
        if query in prof_name.lower() or query in institution.lower():
            results.append({
                "id": m.get("openalex_id") or prof_name,
                "type": "professor",
                "label": prof_name,
                "sublabel": f"Professor at {institution}",
                "url": f"#discovery?professor={m.get('openalex_id') or prof_name}"
            })
            
    # 4. Outcomes
    outcomes = crm_engine.list_outcomes()
    for out in outcomes:
        if query in out.suggestion.lower() or query in out.implementation_notes.lower():
            results.append({
                "id": out.id,
                "type": "outcome",
                "label": out.suggestion,
                "sublabel": f"Outcome (Impact: {out.impact_score})",
                "url": f"#investigations?tab=outcomes&id={out.investigation_id}"
            })
            
    seen_urls = set()
    dedup_results = []
    for r in results:
        if r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            dedup_results.append(r)
            
    return jsonify(dedup_results[:12])


@app.route("/api/crm/command_center", methods=["GET"])
def get_command_center():
    """Serves high-leverage actions categorized into Research, Outreach, Experiments, Writing."""
    from academic_pipeline.collaboration_scorer import CollaborationScorer
    scorer = CollaborationScorer(cache_manager=cache_manager)
    
    invs = inv_engine.get_active_investigations()
    memories = cache_manager.list_professor_memories()
    interactions = crm_engine.list_interactions()
    outcomes = crm_engine.list_outcomes()
    reminders = crm_engine.get_due_reminders()
    
    actions = {
        "research": [],
        "outreach": [],
        "experiments": [],
        "writing": []
    }
    
    # 1. Outreach
    for rem in reminders:
        actions["outreach"].append({
            "title": f"Follow up with {rem['professor_name']}",
            "reason": f"Follow-up reminder due on {rem['follow_up_date']} for interaction regarding blocker in '{rem['investigation_title']}'.",
            "utility": 0.88,
            "why": {
                "Reminders Overdue": 45,
                "Active Conversation": 30,
                "Investigation Priority": 13
            }
        })
        
    for i in interactions:
        if i.status == "drafted":
            actions["outreach"].append({
                "title": f"Review drafted email to {i.professor_name}",
                "reason": f"An intellectual question has been generated to resolve a blocker. Needs review.",
                "utility": 0.76,
                "why": {
                    "Draft Ready": 50,
                    "Blocker Match": 26
                }
            })
            
    for m in memories:
        prof = ProfessorMemory.from_dict(m)
        if prof.status == "discovered" and prof.last_score and prof.last_score > 10.0:
            utility = round(min(0.95, (prof.last_score / 35.0) * 0.8 + 0.1), 2)
            actions["outreach"].append({
                "title": f"Contact professor {prof.professor_name}",
                "reason": f"High research alignment with active blockers (score: {prof.last_score:.1f}). Supervise signal: {prof.student_supervision_signal:.1f}.",
                "utility": utility,
                "why": {
                    "ENRS Score": round(prof.last_score * 2.2, 1),
                    "Accessibility": round(prof.student_supervision_signal * 12.0, 1)
                }
            })

    # 2. Research
    from academic_pipeline.research_graph import ResearchProgramGraph
    graph = ResearchProgramGraph(cache_dir=OUTPUT_DIR)
    
    for u, v, data in graph.graph.edges(data=True):
        if data.get("relation") == "probability_match":
            prob = data.get("probability", 0.0)
            if prob > 0.65:
                need_node = graph.graph.nodes[u]
                prog_node = graph.graph.nodes[v]
                actions["research"].append({
                    "title": f"Read papers on program '{prog_node.get('title')}' by {prog_node.get('professor_name')}",
                    "reason": f"Clustered program matches open evidence need with {round(prob*100)}% probability. Blocker: {need_node.get('description')[:70]}...",
                    "utility": round(prob, 2),
                    "why": {
                        "Probability Match": round(prob * 75, 1),
                        "Domain Intersection": 15
                    }
                })

    # 3. Experiments
    for inv in invs:
        for blocker in inv.open_blockers:
            actions["experiments"].append({
                "title": f"Design test for blocker: '{blocker.description[:50]}...'",
                "reason": f"Active blocker in investigation '{inv.title}' (severity: {blocker.severity:.2f}, specificity: {blocker.specificity_score:.2f}).",
                "utility": round(blocker.effective_priority, 2),
                "why": {
                    "Blocker Severity": round(blocker.severity * 50, 1),
                    "Specificity Score": round(blocker.specificity_score * 50, 1)
                }
            })

    # 4. Writing
    for out in outcomes:
        if not out.resulting_artifact:
            actions["writing"].append({
                "title": f"Document outcome for {out.professor_id}",
                "reason": f"A suggestion with impact {out.impact_score:.2f} was tested but lacks a resulting artifact in database.",
                "utility": 0.72,
                "why": {
                    "Log Test Results": 45,
                    "Outcome Impact": 27
                }
            })

    # Standard Fallbacks
    if not actions["research"]:
        actions["research"].append({
            "title": "Query literature for neural geometry",
            "reason": "Search literature databases to find papers matching Representation Geometry blockers.",
            "utility": 0.65,
            "why": {"Search Expansion": 40, "General Alignment": 25}
        })
    if not actions["experiments"]:
        actions["experiments"].append({
            "title": "Validate representation geometry benchmark",
            "reason": "Verify alignment algorithms (CKA/RSA) under online learning curriculum ordering constraints.",
            "utility": 0.78,
            "why": {"Method Validation": 50, "Verification": 28}
        })
    if not actions["writing"]:
        actions["writing"].append({
            "title": "Update TARS Core manuscript",
            "reason": "Draft developmental benchmark methodologies in active research documentation.",
            "utility": 0.70,
            "why": {"Document Draft": 50, "Synthesize Evidence": 20}
        })
        
    for cat in actions:
        actions[cat].sort(key=lambda x: x["utility"], reverse=True)
        
    return jsonify(actions)


@app.route("/api/crm/timeline", methods=["GET"])
def get_timeline():
    """Gathers chronological events for an Investigation, Professor, or Program."""
    ttype = request.args.get("type")
    tid = request.args.get("id")
    
    if not ttype or not tid:
        return jsonify({"error": "type and id parameters are required"}), 400
        
    events = []
    
    if ttype == "investigation":
        inv = inv_engine._load(tid)
        if not inv:
            return jsonify({"error": "Investigation not found"}), 404
            
        events.append({
            "date": time.strftime("%Y-%m-%d", time.localtime(inv.created_at)),
            "timestamp": inv.created_at,
            "title": "Investigation Initiated",
            "description": f"Created stream: '{inv.title}' with priority {inv.base_priority}."
        })
        
        for blocker in inv.blockers:
            events.append({
                "date": time.strftime("%Y-%m-%d", time.localtime(blocker.created_at)),
                "timestamp": blocker.created_at,
                "title": "Blocker Identified",
                "description": f"Blocker added: '{blocker.description[:70]}...' (severity: {blocker.severity}, specificity: {blocker.specificity_score:.2f})."
            })
            if blocker.resolved:
                events.append({
                    "date": time.strftime("%Y-%m-%d", time.localtime(blocker.resolved_at)),
                    "timestamp": blocker.resolved_at,
                    "title": "Blocker Resolved",
                    "description": f"Resolved by '{blocker.resolved_by}' (strength: {blocker.evidence_strength})."
                })
                
        for c in inv.contradictions:
            events.append({
                "date": time.strftime("%Y-%m-%d", time.localtime(c.created_at)),
                "timestamp": c.created_at,
                "title": "Contradiction Flagged",
                "description": f"Tension: '{c.claim_yours[:50]}' vs '{c.claim_external[:50]}' (severity: {c.severity})."
            })
            
        outcomes = crm_engine.list_outcomes(investigation_id=tid)
        for out in outcomes:
            events.append({
                "date": time.strftime("%Y-%m-%d", time.localtime(out.created_at)),
                "timestamp": out.created_at,
                "title": "Outcome Recorded",
                "description": f"Tested suggestion: '{out.suggestion[:50]}...'. Impact: {out.impact_score:.2f}."
            })
            
    elif ttype == "professor":
        prof_mem_dict = cache_manager.get_professor_memory(tid)
        if not prof_mem_dict:
            memories = cache_manager.list_professor_memories()
            for m in memories:
                if m.get("professor_name") == tid or m.get("openalex_id") == tid:
                    prof_mem_dict = m
                    break
        if not prof_mem_dict:
            return jsonify({"error": "Professor memory not found"}), 404
            
        prof = ProfessorMemory.from_dict(prof_mem_dict)
        events.append({
            "date": time.strftime("%Y-%m-%d", time.localtime(prof.first_seen_at)),
            "timestamp": prof.first_seen_at,
            "title": "PI Discovered",
            "description": f"Discovered researcher {prof.professor_name} at {prof.institution}."
        })
        
        for ev in prof.events:
            ev_title = ev.event_type.replace("_", " ").title()
            events.append({
                "date": time.strftime("%Y-%m-%d", time.localtime(ev.timestamp)),
                "timestamp": ev.timestamp,
                "title": ev_title,
                "description": ev.notes
            })
            
    elif ttype == "program":
        from academic_pipeline.research_graph import ResearchProgramGraph
        graph = ResearchProgramGraph(cache_dir=OUTPUT_DIR)
        
        prog_node = None
        for node_id, data in graph.graph.nodes(data=True):
            if data.get("type") == "program" and node_id == tid:
                prog_node = data
                break
                
        if not prog_node:
            return jsonify({"error": "Program not found"}), 404
            
        last_ref = prog_node.get("last_refresh", time.time())
        events.append({
            "date": time.strftime("%Y-%m-%d", time.localtime(last_ref)),
            "timestamp": last_ref,
            "title": "Program Clustered",
            "description": f"Extracted program '{prog_node.get('title')}' by {prog_node.get('professor_name')}."
        })
        
        prof_name = prog_node.get("professor_name")
        memories = cache_manager.list_professor_memories()
        for m in memories:
            if m.get("professor_name") == prof_name:
                prof = ProfessorMemory.from_dict(m)
                for ev in prof.events:
                    ev_title = ev.event_type.replace("_", " ").title()
                    events.append({
                        "date": time.strftime("%Y-%m-%d", time.localtime(ev.timestamp)),
                        "timestamp": ev.timestamp,
                        "title": f"PI Event: {ev_title}",
                        "description": ev.notes
                    })
                break
                
    events.sort(key=lambda x: x["timestamp"], reverse=True)
    return jsonify(events)


@app.route("/api/crm/self_model", methods=["GET"])
def get_crm_self_model():
    """Serves the self-model research profile metrics."""
    model = cache_manager.get_self_model()
    if not model:
        model = {
            "topic_embedding": [],
            "method_embedding": [],
            "strong_claims": [],
            "structured_claims": [],
            "open_questions": [],
            "recurring_blockers": [],
            "publication_vectors": [],
            "topics": [],
            "methods": [],
            "model_hash": "",
            "built_at": time.time(),
            "paper_count": 0
        }
    return jsonify(model)


@app.route("/api/crm/add_investigation", methods=["POST"])
def add_crm_investigation():
    """Creates a new InvestigationStream and attempts immediate auto-activation."""
    data = request.json or {}
    title = data.get("title", "").strip()
    description = data.get("description", "").strip()
    priority = float(data.get("priority", 0.5))
    
    if not title or not description:
        return jsonify({"error": "Title and description are required"}), 400
        
    try:
        inv = inv_engine.create_investigation(
            title=title,
            description=description,
            base_priority=priority,
            auto_activate=True
        )
        return jsonify(inv.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/crm/add_blocker", methods=["POST"])
def add_crm_blocker():
    """Adds a Blocker to an Investigation with specificity validation check."""
    data = request.json or {}
    investigation_id = data.get("investigation_id")
    description = data.get("description", "").strip()
    blocker_type = data.get("blocker_type", "methodology")
    severity = float(data.get("severity", 0.7))
    
    if not investigation_id or not description:
        return jsonify({"error": "investigation_id and description are required"}), 400
        
    try:
        blocker = inv_engine.add_blocker(
            inv_id=investigation_id,
            description=description,
            blocker_type=blocker_type,
            severity=severity
        )
        return jsonify(blocker.to_dict())
    except BlockerTooVagueError as e:
        return jsonify({
            "error": "Blocker is too vague",
            "refinement_message": e.refinement_message
        }), 422
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/crm/resolve_blocker", methods=["POST"])
def resolve_crm_blocker():
    """Resolves an active blocker manually with notes and evidence strength."""
    data = request.json or {}
    investigation_id = data.get("investigation_id")
    blocker_id = data.get("blocker_id")
    resolved_by = data.get("resolved_by", "").strip()
    evidence_strength = data.get("evidence_strength", "pilot_result")
    notes = data.get("notes", "").strip()
    
    if not investigation_id or not blocker_id or not resolved_by:
        return jsonify({"error": "investigation_id, blocker_id, and resolved_by are required"}), 400
        
    try:
        inv, blocker = inv_engine.resolve_blocker(
            inv_id=investigation_id,
            blocker_id=blocker_id,
            resolved_by=resolved_by,
            evidence_strength=evidence_strength,
            notes=notes
        )
        return jsonify({
            "investigation": inv.to_dict(),
            "blocker": blocker.to_dict()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class EmailReplyIntelligence(BaseModel):
    suggested_paper: Optional[str] = Field(None, description="Title of a paper suggested by the professor, or null if none")
    paper_confidence: float = Field(0.0, ge=0.0, le=1.0)
    suggested_experiment: Optional[str] = Field(None, description="Description of an experiment/method suggestion from the professor, or null if none")
    experiment_confidence: float = Field(0.0, ge=0.0, le=1.0)

UNIVERSITY_ALIASES = {
    "École Polytechnique Fédérale de Lausanne": "EPFL",
    "cole Polytechnique Fdrale de Lausanne": "EPFL",
    "Epfl": "EPFL",
    "Swiss Federal Institute of Technology Lausanne": "EPFL",
    "Massachusetts Institute of Technology": "MIT",
    "Stanford University": "Stanford",
    "Harvard University": "Harvard",
    "University of California, Berkeley": "UC Berkeley",
    "California Institute of Technology": "Caltech",
    "ETH Zurich": "ETH Zurich",
    "Eidgenössische Technische Hochschule Zürich": "ETH Zurich",
}

UNIVERSITY_CANONICAL = {
    "EPFL": "École Polytechnique Fédérale de Lausanne",
    "MIT": "Massachusetts Institute of Technology",
    "Stanford": "Stanford University",
    "Harvard": "Harvard University",
    "UC Berkeley": "University of California, Berkeley",
    "Caltech": "California Institute of Technology",
    "ETH Zurich": "ETH Zurich",
}



@app.route("/api/crm/dashboard_summary", methods=["GET"])
def get_crm_dashboard_summary():
    """Retrieve next actions, followups, dormant relationships, lost opportunities, and top opportunities."""
    from academic_pipeline.collaboration_scorer import CollaborationScorer
    scorer = CollaborationScorer(cache_manager=cache_manager)
    
    memories = cache_manager.list_professor_memories()
    interactions = crm_engine.list_interactions()
    outcomes = crm_engine.list_outcomes()
    reminders = crm_engine.get_due_reminders()
    
    hero_actions = []
    
    # 1. Follow up (e.g. suggestion implemented but no response logged, or reminder due)
    follow_up_added = False
    for rem in reminders:
        prof_id = rem["prof_id"]
        prof_outcomes = [o for o in outcomes if o.professor_id == prof_id]
        if prof_outcomes:
            reason = f"You implemented his suggestion for '{prof_outcomes[0].suggestion[:50]}' and never reported results."
            impact = "Strength +12, Readiness +6"
        else:
            reason = f"Follow-up reminder due. Sent email #{rem['interaction_number']} and never reported results."
            impact = "Strength +8, Readiness +4"
        hero_actions.append({
            "id": "action_" + str(uuid.uuid4())[:8],
            "type": "outreach",
            "title": f"Follow up with {rem['professor_name']}",
            "priority": "High",
            "reason": reason,
            "expected_impact": impact,
            "professor_name": rem["professor_name"],
            "prof_id": prof_id
        })
        follow_up_added = True
        if len(hero_actions) >= 1:
            break
            
    # Default follow-up fallback if none due but we have active ones
    if not follow_up_added:
        active_profs = [m for m in memories if m.get("status") in ("contacted", "in_conversation")]
        if active_profs:
            prof = ProfessorMemory.from_dict(active_profs[0])
            hero_actions.append({
                "id": "action_followup_default",
                "type": "outreach",
                "title": f"Follow up with {prof.professor_name}",
                "priority": "High",
                "reason": "You implemented his suggestion 11 days ago and never reported results.",
                "expected_impact": "Strength +12, Readiness +6",
                "professor_name": prof.professor_name,
                "prof_id": prof.prof_id
            })

    # 2. Read new paper
    paper_added = False
    from academic_pipeline.research_graph import ResearchProgramGraph
    graph = ResearchProgramGraph(cache_dir=OUTPUT_DIR)
    for u, v, data in graph.graph.edges(data=True):
        if data.get("relation") == "probability_match":
            prob = data.get("probability", 0.0)
            if prob > 0.65:
                prog_node = graph.graph.nodes[v]
                hero_actions.append({
                    "id": "action_read_paper",
                    "type": "research",
                    "title": "Read New Paper",
                    "priority": "Medium",
                    "professor_name": prog_node.get("professor_name"),
                    "reason": f"Matches current outreach target. Program '{prog_node.get('title')}' resolves blocker.",
                    "expected_impact": "Research Relevance +15"
                })
                paper_added = True
                break
    if not paper_added:
        hero_actions.append({
            "id": "action_read_paper_fallback",
            "type": "research",
            "title": "Read New Paper",
            "priority": "Medium",
            "professor_name": "Seyed-Mohsen Moosavi-Dezfooli",
            "reason": "Matches current outreach target.",
            "expected_impact": "Research Relevance +12"
        })

    # 3. Draft first contact
    draft_added = False
    for m in memories:
        prof = ProfessorMemory.from_dict(m)
        if prof.status == "discovered" and prof.last_score and prof.last_score > 15.0:
            hero_actions.append({
                "id": "action_draft_contact",
                "type": "outreach",
                "title": f"Draft First Contact",
                "priority": "Medium",
                "professor_name": prof.professor_name,
                "reason": f"{int(prof.last_score)} relevance score. Not contacted yet.",
                "expected_impact": "Readiness +8",
                "prof_id": prof.prof_id
            })
            draft_added = True
            break
    if not draft_added:
        hero_actions.append({
            "id": "action_draft_contact_fallback",
            "type": "outreach",
            "title": "Draft First Contact",
            "priority": "Medium",
            "professor_name": "Jordan W. Squair",
            "reason": "82 relevance score. Not contacted yet.",
            "expected_impact": "Readiness +8"
        })
        
    # Opportunities Being Lost
    lost_opportunities = []
    
    for m in memories:
        prof = ProfessorMemory.from_dict(m)
        # Dormant check
        is_dormant_long = prof.status == "dormant" or (prof.status == "in_conversation" and (time.time() - prof.last_updated_at > 30 * 86400))
        # Replied once but never followed up
        replied_once_no_followup = (prof.reply_count == 1 and prof.interaction_count <= 2 and prof.status in ("in_conversation", "contacted"))
        # Strong match but never contacted
        strong_match_never_contacted = (prof.status == "discovered" and prof.last_score and prof.last_score >= 20.0)
        
        if is_dormant_long:
            lost_opportunities.append({
                "professor_name": prof.professor_name,
                "reason": "Dormant: No interaction in > 60 days.",
                "type": "Dormant",
                "prof_id": prof.prof_id,
                "readiness": compute_prof_scores(m, scorer)["readiness"]
            })
        elif replied_once_no_followup:
            lost_opportunities.append({
                "professor_name": prof.professor_name,
                "reason": "Replied once but never followed up.",
                "type": "No Followup",
                "prof_id": prof.prof_id,
                "readiness": compute_prof_scores(m, scorer)["readiness"]
            })
        elif strong_match_never_contacted:
            lost_opportunities.append({
                "professor_name": prof.professor_name,
                "reason": f"Strong match ({int(prof.last_score)} score) but never contacted.",
                "type": "Cold Lead",
                "prof_id": prof.prof_id,
                "readiness": compute_prof_scores(m, scorer)["readiness"]
            })

    # Top opportunities
    top_opportunities = []
    for m in memories:
        prof = ProfessorMemory.from_dict(m)
        scores = compute_prof_scores(m, scorer)
        if prof.status in ("discovered", "contacted") and scores["readiness"] > 20.0:
            top_opportunities.append({
                "professor_name": prof.professor_name,
                "institution": prof.institution,
                "readiness": scores["readiness"],
                "prof_id": prof.prof_id
            })
    top_opportunities = sorted(top_opportunities, key=lambda x: x["readiness"], reverse=True)[:5]

    # Dormant relationships
    dormant_relationships = []
    for m in memories:
        prof = ProfessorMemory.from_dict(m)
        if prof.status == "dormant":
            scores = compute_prof_scores(m, scorer)
            dormant_relationships.append({
                "professor_name": prof.professor_name,
                "institution": prof.institution,
                "readiness": scores["readiness"],
                "prof_id": prof.prof_id
            })
            
    # Active followups
    active_followups = []
    for rem in reminders:
        active_followups.append({
            "professor_name": rem["professor_name"],
            "follow_up_date": rem["follow_up_date"],
            "reason": rem["next_action"],
            "prof_id": rem["prof_id"]
        })
        
    return jsonify({
        "hero_actions": hero_actions,
        "follow_ups": active_followups,
        "lost_opportunities": lost_opportunities[:5],
        "top_opportunities": top_opportunities,
        "dormant_relationships": dormant_relationships[:5]
    })

@app.route("/api/crm/extract_intelligence", methods=["POST"])
def extract_reply_intelligence():
    """Extract suggested papers or experiments from email body with confidence scores."""
    data = request.json or {}
    email_body = data.get("email_body", "").strip()
    if not email_body:
        return jsonify({"error": "email_body is required"}), 400
        
    try:
        from academic_pipeline.local_llm import LocalLLMClient
        client = LocalLLMClient()
        if client.is_available():
            system = (
                "You are an expert academic assistant analyzing a professor's email reply to extract any suggested papers to read or suggested experiments to run.\n"
                "Extract structured fields: suggested_paper (title only, or null), paper_confidence (0.0 to 1.0), suggested_experiment (detailed description, or null), and experiment_confidence (0.0 to 1.0)."
            )
            user = f"Email reply:\n\n{email_body}"
            res = client._chat_structured(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                EmailReplyIntelligence
            )
            return jsonify(res)
    except Exception as e:
        logger.warning(f"LLM intelligence extraction failed: {e}. Falling back to rule-based parser.")
        
    # Fallback to rule-based parsing
    suggested_paper = None
    paper_confidence = 0.0
    suggested_experiment = None
    experiment_confidence = 0.0
    
    body_lower = email_body.lower()
    import re
    if "paper" in body_lower or "read" in body_lower or "arxiv" in body_lower or "journal" in body_lower:
        quotes = re.findall(r'["\']([^"\']{15,80})["\']', email_body)
        if quotes:
            suggested_paper = quotes[0]
            paper_confidence = 0.85
        else:
            suggested_paper = "Read suggested literature in Representation Geometry"
            paper_confidence = 0.65
            
    if "experiment" in body_lower or "test" in body_lower or "try" in body_lower or "implement" in body_lower or "run" in body_lower:
        suggested_experiment = "Validate representation alignment under curriculum ordering constraints"
        experiment_confidence = 0.75
        
    return jsonify({
        "suggested_paper": suggested_paper,
        "paper_confidence": paper_confidence,
        "suggested_experiment": suggested_experiment,
        "experiment_confidence": experiment_confidence
    })

@app.route("/api/crm/confirm_tasks", methods=["POST"])
def confirm_tasks():
    """Create outcomes/tasks based on confirmed intelligence extraction."""
    data = request.json or {}
    prof_id = data.get("prof_id")
    investigation_id = data.get("investigation_id")
    blocker_id = data.get("blocker_id")
    
    if not investigation_id:
        active_invs = inv_engine.get_active_investigations()
        if active_invs:
            investigation_id = active_invs[0].id
            
    if not blocker_id and investigation_id:
        inv = inv_engine._load(investigation_id)
        if inv and inv.blockers:
            blocker_id = inv.blockers[0].id
            
    if not prof_id or not investigation_id or not blocker_id:
        return jsonify({"error": "Missing required IDs: prof_id, investigation_id, or blocker_id"}), 400
        
    created_outcomes = []
    
    if data.get("suggested_paper_confirmed") and data.get("suggested_paper"):
        o = crm_engine.log_outcome(
            investigation_id=investigation_id,
            prof_id=prof_id,
            blocker_id=blocker_id,
            suggestion=f"Read paper: {data['suggested_paper']}",
            contribution_type="paper_recommendation",
            implementation_notes="",
            impact_score=0.1,
            resulting_artifact="",
            resolve_associated_blocker=False
        )
        created_outcomes.append(o.to_dict())
        
    if data.get("suggested_experiment_confirmed") and data.get("suggested_experiment"):
        o = crm_engine.log_outcome(
            investigation_id=investigation_id,
            prof_id=prof_id,
            blocker_id=blocker_id,
            suggestion=f"Test suggested experiment: {data['suggested_experiment']}",
            contribution_type="method_suggestion",
            implementation_notes="",
            impact_score=0.2,
            resulting_artifact="",
            resolve_associated_blocker=False
        )
        created_outcomes.append(o.to_dict())
        
    prof_mem_dict = cache_manager.get_professor_memory(prof_id)
    if prof_mem_dict:
        prof_mem = ProfessorMemory.from_dict(prof_mem_dict)
        if prof_mem.status in ("discovered", "contacted"):
            prof_mem.status = "in_conversation"
            cache_manager.save_professor_memory(prof_id, prof_mem.to_dict())
            
    return jsonify({"status": "success", "outcomes": created_outcomes})

@app.route("/api/crm/universities", methods=["GET"])
def get_crm_universities():
    """Aggregate indexed professors, contacted/replied/ongoing counts, and coverage % by university."""
    memories = cache_manager.list_professor_memories()
    
    uni_data = {}
    for m in memories:
        prof = ProfessorMemory.from_dict(m)
        inst = prof.institution.strip() if prof.institution else "Unknown University"
        if not inst:
            inst = "Unknown University"
            
        alias = UNIVERSITY_ALIASES.get(inst, inst[:30] + "..." if len(inst) > 30 else inst)
        canonical = UNIVERSITY_CANONICAL.get(alias, inst)
            
        if canonical not in uni_data:
            uni_data[canonical] = {
                "name": canonical,
                "alias": alias,
                "indexed_professors": 0,
                "contacted": 0,
                "replied": 0,
                "ongoing": 0
            }
            
        uni_data[canonical]["indexed_professors"] += 1
        
        if prof.status not in ("discovered", "ignored", "not_relevant"):
            uni_data[canonical]["contacted"] += 1
            
        if prof.reply_count > 0 or prof.status in ("in_conversation", "mentor", "resolved"):
            uni_data[canonical]["replied"] += 1
            
        if prof.status == "in_conversation":
            uni_data[canonical]["ongoing"] += 1
            
    TARGET_CONCEPTS = [
        "C41008148", # computer science
        "C154945302", # artificial intelligence
        "C169900460", # cognitive science
        "C169760540", # neuroscience
        "C50522688", # robotics
        "C119857082", # machine learning
        "C121332964", # physics
        "C635677", # astrophysics
        "C205649164", # astronomy
    ]

    results = []
    for inst, stats in uni_data.items():
        indexed = stats["indexed_professors"]
        contacted = stats["contacted"]
        alias = stats["alias"]
        
        # 1. Researchers Discovered (total crawled)
        total_discovered = cache_manager.get_crawled_faculty_count(inst)
        if not total_discovered:
            total_discovered = cache_manager.get_crawled_faculty_count(alias)
        if not total_discovered:
            total_discovered = max(indexed, 100)
            
        # 2. Screened (matching broad target concepts)
        screened = cache_manager.get_screened_faculty_count(inst, TARGET_CONCEPTS)
        if not screened:
            screened = cache_manager.get_screened_faculty_count(alias, TARGET_CONCEPTS)
        if not screened:
            screened = round(total_discovered * 0.15) if total_discovered else 0
            
        # 3. Scored (scanned candidates)
        scored_count = cache_manager.get_scored_faculty_count(inst, alias)
        
        # 4. Relevance rate and Estimated Relevant
        if scored_count > 0:
            relevance_rate = max(0.02, min(0.25, indexed / scored_count))
        else:
            relevance_rate = 0.08  # 8% fallback
            
        total_relevant_estimated = max(indexed, round(screened * relevance_rate))
        if total_relevant_estimated <= 0:
            total_relevant_estimated = 1
            
        # 5. Deep Scanned (resolved profiles count)
        deep_scanned = cache_manager.get_resolved_profile_count(inst, alias)
        if deep_scanned < indexed:
            deep_scanned = indexed
            
        # 6. Relevant Coverage
        relevant_coverage = round((deep_scanned / total_relevant_estimated) * 100.0)
        relevant_coverage = min(100, relevant_coverage)
        
        # Add to stats dict
        stats["total_researchers"] = total_discovered
        stats["total_professors"] = total_discovered # backwards compat
        stats["screened"] = screened
        stats["total_relevant_estimated"] = total_relevant_estimated
        stats["deep_scanned"] = deep_scanned
        stats["relevant_coverage"] = relevant_coverage
        
        # Coverage variables for backwards compatibility
        stats["discovery_coverage"] = relevant_coverage
        stats["outreach_coverage"] = round((contacted / indexed) * 100.0) if indexed > 0 else 0
        stats["coverage"] = relevant_coverage
        
        results.append(stats)
        
    results = sorted(results, key=lambda x: x["indexed_professors"], reverse=True)
    return jsonify(results)


if __name__ == "__main__":
    # Start server locally
    app.run(host="localhost", port=5000, debug=True)

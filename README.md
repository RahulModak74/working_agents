Working Agents: Deterministic AI Workflows for Production (Please see the ZIP file in Agentic_Ai_Platform directory)
The MCP Problem: Non-deterministic execution. Can't audit. Can't approve. Can't reproduce.
The Solution: Generate workflows once → Review → Execute deterministically forever.

⚠️ The Enterprise Reality Check
python# MCP/LangChain/AutoGen Approach
agent.run("Generate monthly compliance report")
# Run 1: Queries tables A, B, C → 142 records
# Run 2: Queries tables A, C, D → 89 records  
# Run 3: Queries tables A, B → 156 records

Auditor: "Why are these different?"
You: "The AI optimized differently each time..."
Auditor: "🚫 Not compliant. Rejected."
python# Working Agents Approach
workflow = generate_once("compliance_report.json")
# ✅ Compliance reviews and approves

for month in ["Jan", "Feb", "Mar"]:
    execute_workflow(workflow, data=month)
    # Same tables, same order, same logic
    # ✅ Reproducible, auditable, compliant

🎯 Who Needs This?
Not for:

❌ Creative exploration ("surprise me!")
❌ One-off research tasks
❌ Hackathon demos

Perfect for:

✅ Monthly compliance reports (same every time)
✅ Nightly data syncs (deterministic)
✅ Customer onboarding (auditable)
✅ Financial reconciliation (reproducible)
✅ Any regulated workflow that needs pre-approval


💡 The Key Insight
MCP/n8n solve 5% of use cases: Exploratory, creative, unpredictable
Working Agents solves 95% of use cases: Production, regulated, repeatable
You probably need both. Use MCP for exploration. Use Working Agents for everything in production.

🚀 Quick Start (5 Minutes)
bash# 1. Clone and setup
git clone https://github.com/RahulModak74/enhanced_agents.git
cd enhanced_agents
pip install jsonschema

# 2. Configure (get free DeepSeek token from openrouter)
# Edit config.py with your API key

# 3. Run example cybersecurity workflow
python3 main.py --workflow extremely_advanced_cyber_agentic_workflow.json journey.csv

# 4. Run customer journey analysis  
python3 main.py --workflow cust_jour_workflow.json customer_journey.csv (cross_platform_runner.py for cross windows and unix)
That's it. No complex Python code. Just JSON configuration.

🎭 What Makes This Different?
Other Frameworks:
python# LangChain equivalent: 500+ lines of Python
from langchain import Agent, Chain, Memory, Tools
from langchain.callbacks import CustomCallback
# ... 50 more imports

class CustomChain(Chain):
    def _call(self, inputs):
        # 100 lines of orchestration logic
        
class CustomCallback(BaseCallback):
    def on_agent_action(self, action):
        # 50 lines of callback handling
        
# ... 300+ more lines of chain composition, 
# error handling, memory management, etc.
Working Agents:
json[
  {
    "agent": "analyzer",
    "content": "Analyze security data",
    "file": "journey.csv",
    "memory_id": "analysis",
    "output_format": {"type": "json"}
  },
  {
    "agent": "reporter",
    "content": "Generate executive report",
    "readFrom": ["analyzer"],
    "memory_id": "analysis"
  }
]
Same capability. 10x less code. 100% reproducible.

📊 Real-World Examples Included
1. Cybersecurity Analysis (extremely_advanced_cyber_agentic_workflow.json)
Multi-stage threat analysis with:

Session behavior analysis
Pattern recognition across signals
Dynamic response selection
Executive reporting

Run it:
bashpython3 main.py --workflow extremely_advanced_cyber_agentic_workflow.json journey.csv
2. Customer Journey Optimization (cust_jour_workflow.json)
Complete funnel analysis with:

Journey stage analysis
Friction point identification
Segment-specific optimization
ROI calculation

Run it:
bashpython3 main.py --workflow cust_jour_workflow.json customer_journey.csv
3. SAP Data Processing (Community contributed)
Enterprise SAP workflow:
bashpython3 runner_wrapper.py --workflow even_better_sap.json sap_data.csv

🏗️ Architecture: Simple by Design
enhanced_agents/
├── main.py                 # Run workflows
├── config.py              # API configuration
├── memory_manager.py      # SQL-based persistent memory
├── agent.py               # Base agent
├── dynamic_agent.py       # Decision-making agent
├── *.json                 # Workflow definitions
└── COMPONENT/             # Advanced tools (optional)
    ├── sql_tool.py        # Database queries
    ├── http_tool.py       # API calls
    ├── vector_db.py       # Semantic search
    └── planning_tool.py   # Structured reasoning
Core principle: Start simple. Add complexity only when needed.

🔧 Core Features
1. Deterministic Execution
Same workflow → Same results (given same input)
2. SQL-Based Memory
No unnecessary vector embeddings for structured data. Fast, queryable, persistent.
3. JSON Configuration
Define complex workflows without writing Python code.
4. Dynamic Decision Making
Agents can choose paths based on findings, but you control the possible paths.
5. Structured Outputs
JSON schema validation ensures consistent, parseable results.

💼 Enterprise Features

✅ Audit Trail: Every decision is logged
✅ Reproducible: Version control your workflows in git
✅ Approvable: Compliance reviews JSON before production
✅ Testable: Run in staging with test data
✅ Debuggable: Clear error messages, no callback hell


📖 Usage Modes
Interactive CLI:
bashpython3 main.py

> create analyzer deepseek/deepseek-chat
> run analyzer "Analyze security data" file:journey.csv
> memory get analysis
> workflow my_workflow.json
Workflow Execution:
bashpython3 main.py --workflow my_workflow.json data.csv
Universal Runner:
bashpython3 universal_main.py --workflow any_workflow.json any_data.csv

🆚 Framework Comparison
FeatureWorking AgentsLangChainMCPn8nDeterministic✅ Yes❌ No❌ No❌ NoLines of CodeJSON config500+ PythonN/AVisualAudit Trail✅ Built-in⚠️ Manual❌ No⚠️ LogsPre-approval✅ Review JSON❌ Review code❌ Can't⚠️ VisualVersion Control✅ JSON in git⚠️ Code❌ No❌ NoReproducible✅ 100%❌ Varies❌ Varies❌ VariesBest ForProductionPrototypingExplorationMarketing

🎯 When to Use What
┌─────────────────────────────────────────┐
│ Use MCP/LangChain when:                 │
│ • Exploring new problems                │
│ • Creative tasks ("surprise me!")       │
│ • Research and prototyping              │
│ • Unpredictability is acceptable        │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Use Working Agents when:                │
│ • Monthly/quarterly reports             │
│ • Compliance workflows                  │
│ • Data pipelines                        │
│ • Customer onboarding                   │
│ • Financial processes                   │
│ • Any workflow that needs approval      │
└─────────────────────────────────────────┘

🔬 Technical Philosophy
Right Tool for the Job
Most enterprise data lives in:

✅ SQL databases (SAP, CRM, ERP)
✅ CSV files (exports, reports)
✅ Structured formats (JSON, XML)

Not everything needs vector embeddings.
Use vectors only when you actually need semantic search. Use SQL for everything else (it's faster and more precise).
Simplicity Over Complexity

Core system: ~1000 lines of Python
LangChain equivalent: 5000+ lines
Same capabilities for 95% of use cases

Advanced tools in COMPONENT/ available when needed, not forced by default.

🚧 What This Isn't

❌ Not a replacement for exploratory MCP workflows
❌ Not a creative AI assistant
❌ Not for one-off research tasks
❌ Not the "smartest" or most autonomous

This is for production workflows where you need:

Predictability over surprises
Compliance over creativity
Reproducibility over variation


🤝 Contributing
This is open source under SSPL. PRs welcome for:

New example workflows
Additional tool integrations
Documentation improvements
Enterprise feature requests


📚 Learn More

Examples: See *.json workflow files
Components: Read COMPONENT/README.md
Issues: Report bugs or request features
Discussions: Share your use cases


💬 FAQ
Q: Can I use this with OpenAI/Anthropic?
A: Yes, configure in config.py. Currently optimized for DeepSeek (free tier on OpenRouter).
Q: Does this work with my database?
A: Yes, use sql_tool.py in COMPONENT directory. Supports any SQL database.
Q: Can I integrate with my existing tools?
A: Yes, extend http_tool.py or create custom tools in COMPONENT.
Q: Is this production-ready?
A: Yes, but test thoroughly in your environment. This is the philosophy it's built on.
Q: Why not just use LangChain?
A: If you need 500 lines of Python for simple orchestration, go ahead. We prefer JSON.

🎯 Bottom Line
If your AI workflow needs to pass an audit, use Working Agents.
If you're just exploring ideas, use MCP (it's great for that).
Different tools for different jobs. This is the production tool.

📜 License
Server Side Public License (SSPL)

🙏 Acknowledgments
Built out of frustration with non-reproducible AI workflows.
Shared because enterprises deserve better than "trust the AI."

Get started: python3 main.py --workflow extremely_advanced_cyber_agentic_workflow.json journey.csv
Questions? Open an issue.
Success story? Let us know.

The comparison you're really here for:
Feature Parity with LangChain (~75%)
What you get without 500 lines of Python:
CapabilityWorking AgentsLangChainCode RequiredAgent Orchestration✅ JSON config✅ Python code0 vs 100+ linesMemory Management✅ SQL-based✅ Multiple backends0 vs 50+ linesStructured Outputs✅ JSON schema✅ Pydantic0 vs 30+ linesDynamic Routing✅ Built-in✅ Custom chains0 vs 80+ linesTool Integration✅ COMPONENT dir✅ Tool calling10 vs 40+ linesState Persistence✅ Automatic⚠️ Manual0 vs 60+ lines
Total: ~10 lines of JSON vs ~360+ lines of Python for equivalent functionality.

# Universal Browsing Agent

A sophisticated AI-powered web automation system that can accomplish **any web-based task** through intelligent planning and adaptive execution. This agent uses advanced reasoning, chain-of-thought processing, and a beautiful "Liquid Glass" UI to provide transparent, reliable automation for information retrieval, shopping, booking, and more.

## 🌟 Key Features

### Universal Task Completion
- **Any Web Task**: From finding information to shopping, booking flights, checking weather - handles any web-based objective
- **Intelligent Planning**: Analyzes user requests and generates custom execution strategies 
- **Zero Hardcoding**: No task-specific logic - truly generalizable across all domains
- **Google-First Approach**: Always starts with search to find the best sources for each task

### Advanced AI Capabilities
- **Chain-of-Thought Reasoning**: Tracks every step, thought, and decision for full transparency
- **Adaptive Execution**: Learns from failures and tries alternative approaches automatically
- **Action Loop Prevention**: Intelligent detection and recovery from repetitive actions
- **Persistent Retry Logic**: Never gives up on achievable tasks like information retrieval

### Beautiful Liquid Glass UI
- **Stunning Visual Interface**: Modern glass-morphism design with floating animations
- **Interactive Chat**: Ask questions about the process, results, and decisions made
- **Real-time Progress**: See exactly what the agent is thinking and doing
- **Comprehensive Results**: Detailed execution summaries with website information

### Reliability-First Design
- **Conservative Navigation**: Careful scrolling and interaction to avoid missing content
- **Tab Management**: Smart handling of new tabs and page transitions
- **Memory Optimization**: Efficient resource usage for long-running tasks
- **Safety Boundaries**: Automatic stopping at login pages and payment forms

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Chrome/Chromium browser installed
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd andera_takehome_submit
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Copy the example environment file
   cp env_example.txt .env
   
   # Edit .env and add your OpenAI API key
   OPENAI_API_KEY=sk-your-openai-api-key-here
   ```

4. **Run your first task**
   ```python
   from main import GeneralizableBrowsingAgent
   
   # Initialize the agent
   agent = GeneralizableBrowsingAgent(
       model="gpt-4o",
       headless=False,  # Set to True to hide browser
       enable_liquid_glass=True
   )
   
   # Execute any web task
   result = await agent.execute_any_task("Find the weather in San Francisco")
   print(result)
   ```

## 📖 Usage Examples

### Information Retrieval
```python
# Weather information
await agent.execute_any_task("What's the weather like in Tokyo today?")

# Business information  
await agent.execute_any_task("Find the phone number and hours for the Apple Store in downtown Seattle")

# Current events
await agent.execute_any_task("What are the latest news headlines about AI technology?")

# Research queries
await agent.execute_any_task("Find statistics about renewable energy adoption in 2024")
```

### Shopping Tasks
```python
# Product search and cart addition
await agent.execute_any_task("Find and add a Wilson NBA basketball to cart")

# Price comparison
await agent.execute_any_task("Compare prices for iPhone 15 Pro across different retailers")

# Product research
await agent.execute_any_task("Find the best rated wireless headphones under $200")
```

### Travel and Booking
```python
# Flight search
await agent.execute_any_task("Find flights from San Francisco to New York for next Friday")

# Hotel booking
await agent.execute_any_task("Find available hotels in Paris for next weekend")

# Restaurant reservations
await agent.execute_any_task("Check availability at popular restaurants in downtown Chicago")
```

### Complex Multi-Step Tasks
```python
# Research and planning
await agent.execute_any_task("Plan a 3-day itinerary for visiting Washington DC, including museums and restaurants")

# Comparative analysis
await agent.execute_any_task("Compare the features and prices of the top 3 project management software tools")
```

## 🏗️ Project Architecture

### Core Components

#### 1. Task Planner Agent (`main.py`)
- Analyzes user queries and generates execution plans
- Creates task schemas with success criteria
- No hardcoded task-specific logic

#### 2. Universal Browsing Agent (`main.py`)
- Executes tasks using intelligent browser automation
- Implements chain-of-thought reasoning
- Handles failures and retry logic

#### 3. Liquid Glass UI (`liquid_glass_server.py`, `liquid_glass_ui.html`)
- Beautiful glass-morphism interface
- Real-time task monitoring
- Interactive chat for post-task questions

#### 4. Configuration System (`config.py`)
- Flexible settings for all aspects of the system
- Reliability vs speed trade-offs
- Memory optimization options

### Key Architectural Decisions

**Universal Approach**: No task-specific code paths. The system analyzes any request and generates appropriate strategies dynamically.

**Chain of Thought**: Every action, thought, and decision is tracked for transparency and debugging.

**Reliability First**: Conservative navigation, aggressive retry logic, and comprehensive error handling prioritize success over speed.

**Safety Boundaries**: Automatic detection and stopping at login pages, payment forms, and other sensitive areas.

## ⚙️ Configuration

The system is highly configurable through `config.py`:

### Browser Settings
```python
HEADLESS_MODE = False           # Show/hide browser window
WAIT_BETWEEN_ACTIONS = 1.0      # Delay between actions
BROWSER_WIDTH = 1920            # Window dimensions
BROWSER_HEIGHT = 1080
```

### Agent Behavior
```python
MAX_STEPS_DEFAULT = 50          # Default step limit
RELIABILITY_MODE = True         # Enable all reliability features
USE_VISION = True              # Screenshot analysis
USE_THINKING = True            # AI reasoning
```

### Memory Optimization
```python
MEMORY_OPTIMIZATION = True      # Enable memory optimizations
MAX_HISTORY_ITEMS = 15         # Limit history size
MAX_PAGE_SIZE_MB = 50          # Skip large pages
```

### Scroll Behavior
```python
SCROLL_SETTINGS = {
    "conservative_scrolling": True,
    "default_scroll_pages": 0.3,    # 30% viewport scrolling
    "overlap_scrolling": True,      # Content overlap
}
```

## 📁 Project Structure

```
andera_takehome_submit/
├── main.py                    # Core agent implementation
├── config.py                  # Configuration settings
├── liquid_glass_server.py     # UI server and API
├── liquid_glass_ui.html       # Beautiful web interface
├── requirements.txt           # Python dependencies
├── env_example.txt           # Environment variables template
├── ARCHITECTURE.md           # Detailed architecture documentation
└── browser_data/             # Persistent browser sessions (created on first run)
```

## 🎨 Liquid Glass UI

The Liquid Glass UI provides a stunning visual interface for monitoring and interacting with the agent:

### Features
- **Glass Morphism Design**: Modern translucent interface with smooth animations
- **3D Flip Animation**: Hover to flip between task summary and interactive chat
- **Chain of Thought Display**: See exactly what the agent is thinking
- **Interactive Chat**: Ask questions about the process and results
- **Real-time Updates**: Live progress monitoring during task execution

### Usage
The UI automatically opens when tasks complete. You can:
1. **View Task Summary**: See what was accomplished and how
2. **Hover to Chat**: Flip the interface to ask questions
3. **Get Details**: Ask about websites visited, steps taken, or results found

## 🛡️ Safety Features

### Automatic Safety Boundaries
- **Private Information Tasks**: Stops at login pages (never enters credentials)
- **Purchase Tasks**: Stops at cart confirmation (never completes payments)
- **Travel Booking**: Stops at booking forms (never enters payment details)

### Error Prevention
- **Action Loop Detection**: Prevents infinite repetition of failed actions
- **Tab Management**: Smart handling of new tabs and page transitions
- **Memory Limits**: Prevents resource exhaustion on long tasks
- **Graceful Degradation**: Tries alternative approaches when primary methods fail

## 🔧 Advanced Usage

### Custom Model Configuration
```python
agent = GeneralizableBrowsingAgent(
    model="gpt-4o-mini",          # Use faster/cheaper model
    headless=True,                # Run without GUI
    enable_liquid_glass=False     # Disable UI for automation
)
```

### Batch Task Processing
```python
tasks = [
    "Find weather in Boston",
    "Check Apple stock price", 
    "Search for pizza restaurants nearby"
]

results = []
for task in tasks:
    result = await agent.execute_any_task(task)
    results.append(result)
```

### Custom Configuration
```python
from config import get_config

# Modify settings
config = get_config()
config["max_steps"] = 100
config["wait_between_actions"] = 0.5

# Use custom config
agent = GeneralizableBrowsingAgent(**config)
```

## 🤝 Contributing

This project welcomes contributions! Here are some areas where help would be appreciated:

### Potential Improvements
- **Enhanced Vision**: Better screenshot analysis and element detection
- **Multi-Agent Coordination**: Parallel task execution across multiple agents  
- **Predictive Caching**: Pre-loading likely next steps for faster execution
- **Additional LLM Support**: Integration with more AI providers
- **Enterprise Features**: Audit logging, SSO integration, API access

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📋 System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- Chrome/Chromium browser
- Internet connection
- OpenAI API access

### Recommended
- Python 3.11+
- 8GB+ RAM
- High-speed internet
- SSD storage for browser data

## 🐛 Troubleshooting

### Common Issues

**Browser fails to start**
- Ensure Chrome/Chromium is installed
- Check if another instance is running
- Try deleting `browser_data/` folder

**API errors**
- Verify OpenAI API key in `.env` file
- Check API quota and billing
- Ensure stable internet connection

**Tasks get stuck**
- The agent includes automatic loop detection
- Long tasks will eventually timeout and provide partial results
- Check browser memory usage if running many tasks

**Liquid Glass UI doesn't open**
- Check that port 5000 is available
- Try manually visiting `http://localhost:5000`
- Disable firewall/antivirus temporarily

## 📄 License

This project is provided for educational and evaluation purposes. Please ensure compliance with website terms of service and applicable laws when using this automation tool.

## 🙏 Acknowledgments

- Built on the excellent [browser-use](https://github.com/browser-use/browser-use) framework
- Powered by OpenAI's GPT models
- UI inspired by modern glass-morphism design trends
- Architecture influenced by reliable automation principles

---

**Ready to automate any web task?** Start with the quick start guide above and experience the power of universal browsing automation! 🚀 
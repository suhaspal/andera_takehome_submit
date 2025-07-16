#!/usr/bin/env python3
"""
Liquid Glass UI Server
Serves the liquid glass interface and provides API endpoints for task data.
"""

import json
import threading
import time
from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
import webbrowser
import os

app = Flask(__name__)
CORS(app)

# Global variable to store task data
current_task_data = {
    "description": "Task completed successfully",
    "website": "example.com",
    "websiteDescription": "Website visited during task execution",
    "originalQuery": "Sample query",
    "status": "completed",
    "executionTime": "0 seconds",
    "stepsCompleted": 0,
    "taskType": "general",
    "websitesVisited": ["example.com"],
    "actionDetails": [],
    "searchStrategy": "Google search approach",
    "successCriteria": [],
    "fullResult": "Task completed successfully"
}

def read_html_file():
    """Read the liquid glass HTML file"""
    try:
        with open('liquid_glass_ui.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <!DOCTYPE html>
        <html>
        <head><title>Error</title></head>
        <body>
            <h1>Error: liquid_glass_ui.html not found</h1>
            <p>Please make sure the liquid_glass_ui.html file exists in the same directory.</p>
        </body>
        </html>
        """

@app.route('/')
def index():
    """Serve the liquid glass UI"""
    html_content = read_html_file()
    return html_content

@app.route('/api/task-data', methods=['GET'])
def get_task_data():
    """Get current task data"""
    return jsonify(current_task_data)

@app.route('/api/task-data', methods=['POST'])
def update_task_data():
    """Update task data from the browsing agent"""
    global current_task_data
    try:
        data = request.get_json()
        if data:
            current_task_data.update(data)
            return jsonify({"status": "success", "message": "Task data updated"})
        else:
            return jsonify({"status": "error", "message": "No data provided"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages for the liquid glass interface"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        task_data = data.get('taskData', current_task_data)
        
        if not user_message:
            return jsonify({"status": "error", "message": "No message provided"}), 400
        
        # Use the task data from the request if available, otherwise use global
        task_data_to_use = task_data if task_data else current_task_data
        
        # Generate enhanced response based on task data
        response = generate_chat_response(user_message, task_data_to_use)
        
        return jsonify({
            "status": "success",
            "response": response,
            "timestamp": time.time()
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def generate_chat_response(user_message, task_data):
    """Generate contextual responses using ChatGPT API - extract relevant info and format proper answers"""
    
    # Extract available context from task data
    context = {
        'task_type': task_data.get('taskType', 'general'),
        'websites_visited': task_data.get('websitesVisited', []),
        'action_details': task_data.get('actionDetails', []),
        'product_details': task_data.get('productDetails', {}),
        'original_query': task_data.get('originalQuery', ''),
        'website': task_data.get('website', 'Unknown'),
        'description': task_data.get('description', ''),
        'full_result': task_data.get('fullResult', ''),
        'steps_completed': task_data.get('stepsCompleted', 0),
        'execution_time': task_data.get('executionTime', 'Unknown')
    }
    
    # Create context prompt for ChatGPT
    context_prompt = f"""
You are a helpful assistant that completed a web automation task. Here's the context of what you accomplished:

TASK COMPLETED: {context['original_query']}
WEBSITE USED: {context['website']}
DESCRIPTION: {context['description']}
STEPS COMPLETED: {context['steps_completed']}
EXECUTION TIME: {context['execution_time']}

PRODUCT DETAILS (if shopping task):
{context['product_details']}

FULL TASK RESULT:
{context['full_result'][:500]}...

The user is asking: "{user_message}"

Rules for your response:
1. Answer ONLY what the user asked - be specific and direct
2. Use the actual data from the task context above
3. If it's about price/cost, give the exact price from product details
4. If it's about website, mention the specific site used
5. If it's about the product, give specific product details
6. Keep responses conversational but informative
7. Use HTML formatting (<strong>, <br>) for better readability
8. Don't give generic answers - use the actual context data

Respond naturally as if you just completed this task for the user.
"""
    
    try:
        import openai
        import os
        
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": context_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=200,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        # Fallback to context-aware response if OpenAI fails
        return generate_fallback_response(user_message, context)

def generate_fallback_response(user_message, context):
    """Fallback response using context data if OpenAI API fails"""
    message = user_message.lower().strip()
    
    # Extract specific info from context
    if any(word in message for word in ['price', 'cost', 'how much', 'money']):
        if context['product_details'] and 'price' in context['product_details']:
            return f"The price was <strong>{context['product_details']['price']}</strong> for the {context['product_details'].get('name', 'product')}."
        else:
            return "I found pricing information during the task, but the specific price details aren't available in the extracted data."
    
    elif any(word in message for word in ['website', 'site', 'where']):
        return f"I completed the task on <strong>{context['website']}</strong>."
    
    elif any(word in message for word in ['product', 'item', 'basketball']):
        if context['product_details']:
            name = context['product_details'].get('name', 'Unknown product')
            price = context['product_details'].get('price', 'Price not available')
            return f"I found: <strong>{name}</strong> for <strong>{price}</strong>."
        else:
            return f"I found product information during the task on {context['website']}."
    
    else:
        return f"I completed your request: {context['original_query']} on {context['website']}. {context['description']}"



def generate_process_explanation(task_data):
    """Generate detailed process explanation based on task type"""
    task_type = task_data.get('taskType', 'general')
    website = task_data['website']
    action_details = task_data.get('actionDetails', [])
    websites_visited = task_data.get('websitesVisited', [])
    
    # If we have detailed action data, use it for a more specific explanation
    if action_details:
        search_actions = [action for action in action_details if action['type'] == 'search']
        navigate_actions = [action for action in action_details if action['type'] == 'navigate']
        click_actions = [action for action in action_details if action['type'] == 'click']
        input_actions = [action for action in action_details if action['type'] == 'input']
        
        explanation = f"Here's exactly how I completed your task:<br><br>"
        
        if search_actions:
            explanation += f"<strong>1. Search Phase:</strong><br>"
            for action in search_actions:
                explanation += f"• {action['details']}<br>"
            explanation += "<br>"
        
        if navigate_actions:
            explanation += f"<strong>2. Navigation Phase:</strong><br>"
            for action in navigate_actions[:3]:  # Show first 3 navigation actions
                explanation += f"• {action['details']}<br>"
            if len(navigate_actions) > 3:
                explanation += f"• ... and {len(navigate_actions) - 3} more navigation steps<br>"
            explanation += "<br>"
        
        if input_actions:
            explanation += f"<strong>3. Data Entry Phase:</strong><br>"
            for action in input_actions:
                explanation += f"• {action['details']}<br>"
            explanation += "<br>"
        
        if click_actions:
            explanation += f"<strong>4. Interaction Phase:</strong><br>"
            explanation += f"• Performed {len(click_actions)} click actions to navigate and interact with the website<br><br>"
        
        if websites_visited:
            explanation += f"<strong>Websites Visited:</strong> {', '.join(websites_visited)}<br>"
            explanation += f"<strong>Final Website:</strong> {website}"
        
        return explanation
    
    # Fallback to generic explanations
    if task_type in ['flight', 'travel']:
        return f"For flight booking, I follow this process:<br>1. <strong>Google search</strong> to find the best flight booking sites<br>2. <strong>Selected {website}</strong> for its comprehensive options<br>3. <strong>Entered travel details</strong> including dates and destinations<br>4. <strong>Searched for flights</strong> and compared available options<br>5. <strong>Selected optimal flight</strong> based on your criteria<br>6. <strong>Proceeded to booking</strong> up to the payment stage (stopped for safety)"
    
    elif task_type == 'weather':
        return f"For weather information, I follow this process:<br>1. <strong>Google search</strong> to find reliable weather sources<br>2. <strong>Selected {website}</strong> for its accuracy and detail<br>3. <strong>Navigated to location-specific page</strong><br>4. <strong>Extracted current conditions</strong> and relevant weather data<br>5. <strong>Verified information</strong> was current and accurate"
    
    elif task_type == 'shopping':
        return f"For shopping tasks, I follow this process:<br>1. <strong>Google search</strong> to find the best shopping options<br>2. <strong>Selected {website}</strong> for its inventory and pricing<br>3. <strong>Searched for requested items</strong><br>4. <strong>Compared options</strong> and selected the best matches<br>5. <strong>Added items to cart</strong> (stopped before actual purchase for safety)"
    
    else:
        return f"I follow a systematic process:<br>1. <strong>Google search</strong> to find the best sources<br>2. <strong>Selected {website}</strong> as the most reliable option<br>3. <strong>Navigated through the site</strong> systematically<br>4. <strong>Located relevant information</strong> or completed the requested action<br>5. <strong>Extracted or processed</strong> the final results"

class LiquidGlassServer:
    """Class to manage the liquid glass server"""
    
    def __init__(self, port=5000):
        self.port = port
        self.server_thread = None
        self.is_running = False
    
    def start(self):
        """Start the server in a separate thread"""
        if not self.is_running:
            self.server_thread = threading.Thread(target=self._run_server, daemon=True)
            self.server_thread.start()
            self.is_running = True
            time.sleep(1)  # Give server time to start
            print(f"🌐 Liquid Glass UI server started on http://localhost:{self.port}")
    
    def _run_server(self):
        """Run the Flask server"""
        app.run(host='0.0.0.0', port=self.port, debug=False, use_reloader=False)
    
    def update_task_data(self, task_data):
        """Update the task data that will be displayed in the UI"""
        global current_task_data
        current_task_data.update(task_data)
        print(f"📝 Task data updated: {task_data.get('description', 'No description')}")
    
    def show_ui(self):
        """Open the liquid glass UI in the default browser"""
        url = f"http://localhost:{self.port}"
        try:
            webbrowser.open(url)
            print(f"🚀 Liquid Glass UI opened: {url}")
        except Exception as e:
            print(f"❌ Could not open browser: {e}")
            print(f"🔗 Please manually visit: {url}")
    
    def stop(self):
        """Stop the server"""
        self.is_running = False
        print("🛑 Liquid Glass UI server stopped")

# Global server instance
liquid_glass_server = LiquidGlassServer()

def start_liquid_glass_ui():
    """Start the liquid glass UI server"""
    liquid_glass_server.start()

def show_liquid_glass_ui(task_data):
    """Show the liquid glass UI with task data"""
    liquid_glass_server.update_task_data(task_data)
    liquid_glass_server.show_ui()

def update_liquid_glass_data(task_data):
    """Update task data in the liquid glass UI"""
    liquid_glass_server.update_task_data(task_data)

if __name__ == '__main__':
    # Start the server
    liquid_glass_server.start()
    
    # Example task data
    example_task_data = {
        "description": "I successfully found the weather information for Fremont, CA. The temperature is 72°F with partly cloudy conditions.",
        "website": "weather.com/weather/today/l/Fremont+CA",
        "websiteDescription": "Comprehensive weather portal with current conditions and forecasts",
        "originalQuery": "what is weather like in Fremont, CA",
        "status": "completed",
        "executionTime": "45 seconds",
        "stepsCompleted": 8
    }
    
    # Update with example data
    liquid_glass_server.update_task_data(example_task_data)
    
    # Open the UI
    liquid_glass_server.show_ui()
    
    print("🌟 Liquid Glass UI is running!")
    print("🔗 Visit http://localhost:5000 to view the interface")
    print("💬 The UI includes an interactive chat interface")
    print("📊 Task data can be updated via the API endpoints")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        # Keep the server running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down Liquid Glass UI server...")
        liquid_glass_server.stop() 
"""Main app module to run Flask app"""
import traceback
from flask import Flask, request, render_template, jsonify
from main import graph

app = Flask(__name__)


@app.route('/health')
def health():
    """Health check endpoint for Docker"""
    return {'status': 'healthy'}, 200

@app.route("/")
def index():
    """Index page"""
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query():
    """Query endpoint for user's input"""
    try:
        # Check if request has JSON data
        if not request.json:
            return jsonify({"error": "No JSON data received"}), 400

        if "question" not in request.json:
            return jsonify({"error": "No 'question' field in JSON data"}), 400

        user_input = request.json["question"]
        print(f"Processing query: {user_input}")

        final_answer = None

        # Iterate over stream and get final generated answer.
        for step in graph.stream({"messages": [{"type": "human", "content": user_input}]},
                                 stream_mode="values"):

            last_message = step["messages"][-1]
            print("\n--- GRAPH STEP ---")
            last_message.pretty_print()

            # The final response is the last AI message that does NOT have tool calls.
            last_message = step["messages"][-1]
            if last_message.type == "ai" and not getattr(last_message, "tool_calls", []):
                final_answer = last_message.content

        # After the stream is fully processed, return the final answer.
        if final_answer:
            print(f"Returning content: {final_answer}")
            return jsonify({"answer": final_answer})
        else:
            return jsonify({"error": "The graph did not produce a final answer."}), 500

    except Exception as e:
        print(f"Error in query: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Query processing error: {str(e)}"}), 500


if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)

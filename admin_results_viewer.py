"""
Standalone Admin Results Viewer
Run this from the streamlit folder: python admin/admin_results_viewer.py
Or from streamlit/admin folder: python admin_results_viewer.py
"""

import json
import pandas as pd
from datetime import datetime
import os

# Determine the correct path based on where script is run from
if os.path.basename(os.getcwd()) == "admin":
    DATA_PATH = "../data/"
elif os.path.basename(os.getcwd()) == "streamlit":
    DATA_PATH = "data/"
else:
    DATA_PATH = "streamlit/data/"

def view_test_results():
    """Display all test results in readable format"""
    try:
        filepath = os.path.join(DATA_PATH, "test_results.json")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = data.get("results", [])
        
        if not results:
            print("No test data available.")
            return
        
        print("="*80)
        print("📊 TEST RESULTS SUMMARY")
        print("="*80)
        
        # Group by user
        users = {}
        for result in results:
            user_id = result["user_id"]
            if user_id not in users:
                users[user_id] = {"pre": None, "post": None, "info": None}
            
            users[user_id][result["test_type"]] = result
            if result["test_type"] == "pre":
                users[user_id]["info"] = result["participant_info"]
        
        print(f"\nTotal Unique Users: {len(users)}")
        print(f"Total Test Submissions: {len(results)}")
        
        # Summary table
        print("\n" + "="*80)
        print("USER SUMMARY")
        print("="*80)
        
        summary_data = []
        
        for user_id, tests in users.items():
            pre_score = tests["pre"]["scores"]["Overall"] if tests["pre"] else None
            post_score = tests["post"]["scores"]["Overall"] if tests["post"] else None
            improvement = (post_score - pre_score) if (pre_score and post_score) else None
            
            info = tests["info"] if tests["info"] else {}
            
            summary_data.append({
                "User ID": user_id[-8:],  # Last 8 chars for brevity
                "Age": info.get("age", "N/A"),
                "Education": info.get("education", "N/A"),
                "Pre-Test": f"{pre_score:.1f}%" if pre_score else "Not taken",
                "Post-Test": f"{post_score:.1f}%" if post_score else "Not taken",
                "Improvement": f"+{improvement:.1f}%" if improvement else "N/A"
            })
        
        # Create DataFrame for nice display
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
        
        # Detailed results
        print("\n" + "="*80)
        print("DETAILED RESULTS")
        print("="*80)
        
        for user_id, tests in users.items():
            print(f"\n{'='*80}")
            print(f"USER ID: {user_id}")
            print(f"{'='*80}")
            
            if tests["info"]:
                info = tests["info"]
                print(f"\n📋 Participant Information:")
                print(f"   Age: {info.get('age')}")
                print(f"   Gender: {info.get('gender')}")
                print(f"   Education: {info.get('education')}")
                print(f"   Occupation: {info.get('occupation')}")
            
            for test_type in ["pre", "post"]:
                if tests[test_type]:
                    result = tests[test_type]
                    print(f"\n{'📝 PRE-TEST' if test_type == 'pre' else '📋 POST-TEST'} RESULTS:")
                    print(f"   Timestamp: {result['timestamp']}")
                    
                    print(f"\n   Scores by Category:")
                    for category, score in result["scores"].items():
                        print(f"      {category:25} : {score:6.1f}%")
                    
                    print(f"\n   Detailed Responses:")
                    for resp in result["responses"]:
                        print(f"      Q: {resp['question'][:60]}...")
                        print(f"      A: {resp['response']} (Score: {resp['score']})")
                        print()
        
        # Export to CSV
        df.to_csv("data/test_results_summary.csv", index=False)
        print("\n✅ Summary exported to: data/test_results_summary.csv")
        
    except FileNotFoundError:
        print("❌ Error: data/test_results.json file not found!")
        print("💡 Tip: Run the Streamlit app first and complete a test to generate data.")
    except Exception as e:
        print(f"❌ Error: {e}")

def view_feedback():
    """Display all user feedback"""
    try:
        with open("data/user_feedback.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        feedbacks = data.get("feedback", [])
        
        if not feedbacks:
            print("No feedback data available.")
            return
        
        print("\n" + "="*80)
        print("💬 USER FEEDBACK")
        print("="*80)
        
        helpful = [f for f in feedbacks if f["rating"] == "helpful"]
        not_helpful = [f for f in feedbacks if f["rating"] == "not_helpful"]
        
        print(f"\nTotal Feedback: {len(feedbacks)}")
        print(f"👍 Helpful: {len(helpful)} ({len(helpful)/len(feedbacks)*100:.1f}%)")
        print(f"👎 Not Helpful: {len(not_helpful)} ({len(not_helpful)/len(feedbacks)*100:.1f}%)")
        
        print("\n" + "="*80)
        print("DETAILED FEEDBACK")
        print("="*80)
        
        for i, fb in enumerate(feedbacks, 1):
            print(f"\n{'-'*80}")
            print(f"Feedback #{i}")
            print(f"{'-'*80}")
            print(f"User ID: {fb['user_id']}")
            print(f"Timestamp: {fb['timestamp']}")
            print(f"Rating: {fb['rating'].upper()}")
            print(f"\nQuestion: {fb['question']}")
            print(f"\nAnswer Preview: {fb['answer'][:200]}...")
            print(f"\nSources Used: {fb['sources_count']}")
        
        # Export to CSV
        feedback_data = []
        for fb in feedbacks:
            feedback_data.append({
                "User ID": fb["user_id"][-8:],
                "Timestamp": fb["timestamp"],
                "Rating": fb["rating"],
                "Question": fb["question"],
                "Sources": fb["sources_count"]
            })
        
        df = pd.DataFrame(feedback_data)
        df.to_csv("data/feedback_summary.csv", index=False)
        print("\n✅ Feedback exported to: data/feedback_summary.csv")
        
    except FileNotFoundError:
        print("❌ Error: data/user_feedback.json file not found!")
        print("💡 Tip: Run the Streamlit app and provide feedback to generate data.")
    except Exception as e:
        print(f"❌ Error: {e}")

def calculate_statistics():
    """Calculate and display statistics"""
    try:
        with open("data/test_results.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = data.get("results", [])
        
        if not results:
            print("No data available for statistics.")
            return
        
        print("\n" + "="*80)
        print("📈 STATISTICS & ANALYTICS")
        print("="*80)
        
        # Group by user for improvement calculation
        users = {}
        for result in results:
            user_id = result["user_id"]
            if user_id not in users:
                users[user_id] = {}
            users[user_id][result["test_type"]] = result["scores"]["Overall"]
        
        # Calculate improvements
        improvements = []
        for user_id, scores in users.items():
            if "pre" in scores and "post" in scores:
                improvement = scores["post"] - scores["pre"]
                improvements.append({
                    "user_id": user_id,
                    "pre": scores["pre"],
                    "post": scores["post"],
                    "improvement": improvement
                })
        
        if improvements:
            print(f"\n✅ Users with Complete Tests: {len(improvements)}")
            
            avg_pre = sum(i["pre"] for i in improvements) / len(improvements)
            avg_post = sum(i["post"] for i in improvements) / len(improvements)
            avg_improvement = sum(i["improvement"] for i in improvements) / len(improvements)
            
            print(f"\nAverage Pre-Test Score: {avg_pre:.1f}%")
            print(f"Average Post-Test Score: {avg_post:.1f}%")
            print(f"Average Improvement: {avg_improvement:+.1f}%")
            
            improved = sum(1 for i in improvements if i["improvement"] > 0)
            declined = sum(1 for i in improvements if i["improvement"] < 0)
            same = sum(1 for i in improvements if i["improvement"] == 0)
            
            print(f"\n📊 Improvement Distribution:")
            print(f"   Improved: {improved} ({improved/len(improvements)*100:.1f}%)")
            print(f"   Declined: {declined} ({declined/len(improvements)*100:.1f}%)")
            print(f"   No Change: {same} ({same/len(improvements)*100:.1f}%)")
            
            # Best and worst performers
            improvements_sorted = sorted(improvements, key=lambda x: x["improvement"], reverse=True)
            
            print(f"\n🏆 Top 3 Improvements:")
            for i, imp in enumerate(improvements_sorted[:3], 1):
                print(f"   {i}. User {imp['user_id'][-8:]}: {imp['pre']:.1f}% → {imp['post']:.1f}% ({imp['improvement']:+.1f}%)")
            
            if len(improvements_sorted) >= 3:
                print(f"\n⚠️  Bottom 3 Improvements:")
                for i, imp in enumerate(improvements_sorted[-3:], 1):
                    print(f"   {i}. User {imp['user_id'][-8:]}: {imp['pre']:.1f}% → {imp['post']:.1f}% ({imp['improvement']:+.1f}%)")
        else:
            print("\n⚠️  No complete pre/post test pairs available yet.")
        
        # Category breakdown
        print(f"\n📊 Category Performance:")
        categories = ["Financial Knowledge", "Financial Behavior", "Financial Confidence", "Financial Attitudes"]
        
        for category in categories:
            cat_improvements = []
            for result in results:
                if result["test_type"] == "post":
                    user_id = result["user_id"]
                    # Find matching pre-test
                    pre_result = next((r for r in results if r["user_id"] == user_id and r["test_type"] == "pre"), None)
                    if pre_result:
                        improvement = result["scores"][category] - pre_result["scores"][category]
                        cat_improvements.append(improvement)
            
            if cat_improvements:
                avg = sum(cat_improvements) / len(cat_improvements)
                print(f"   {category:25} : {avg:+6.1f}%")
        
    except FileNotFoundError:
        print("❌ Error: data/test_results.json file not found!")
        print("💡 Tip: Complete both pre-test and post-test to see statistics.")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main menu"""
    print("\n" + "="*80)
    print("👨‍💼 ADMIN RESULTS VIEWER")
    print("="*80)
    
    while True:
        print("\n📋 Menu:")
        print("1. View Test Results")
        print("2. View User Feedback")
        print("3. View Statistics & Analytics")
        print("4. View All")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            view_test_results()
        elif choice == "2":
            view_feedback()
        elif choice == "3":
            calculate_statistics()
        elif choice == "4":
            view_test_results()
            view_feedback()
            calculate_statistics()
        elif choice == "5":
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid option. Please try again.")

if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        print("⚠️  pandas not installed. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "pandas"])
        import pandas as pd
    
    main()
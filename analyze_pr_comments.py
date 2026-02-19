import json
import sys

# Force stdout to use utf-8
sys.stdout.reconfigure(encoding='utf-8')

try:
    with open('pr_comments.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total Comments: {len(data.get('comments', []))}")
    print(f"Total Reviews: {len(data.get('reviews', []))}")
    print("-" * 50)

    for i, comment in enumerate(data.get('comments', [])):
        print(f"COMMENT #{i+1} by {comment['author']['login']} ({comment['createdAt']})")
        print(comment['body'])
        print("-" * 50)

    for i, review in enumerate(data.get('reviews', [])):
        print(f"REVIEW #{i+1} by {review['author']['login']} ({review['submittedAt']})")
        print(review['body'])
        print("-" * 50)

except Exception as e:
    print(f"Error parsing JSON: {e}")

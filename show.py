import tkinter as tk
import pandas as pd
from random import choice, randint
from v3 import UserTraitPredictor  # Ensure your class is saved here

# Initialize predictor and load trained model
predictor = UserTraitPredictor()
predictor.load_model('user_trait_model.pkl')

# Define pools for random generation
locations = ['Oslo', 'Bergen', 'Trondheim']
hobby_pool = ['music', 'gaming', 'pets', 'reading', 'travel', 'tech']
traits = ['introvert', 'extrovert', 'dreamer', 'analytical', 'impulsive']

# State trackers
step = 0
current_user = None
prediction = None
campaign = None

# GUI setup
root = tk.Tk()
root.title("Persona Campaign Demo")
root.geometry("850x550")

text_box = tk.Text(root, font=("Courier", 12), wrap='word')
text_box.pack(padx=20, pady=20, fill='both', expand=True)


def generate_random_user():
    return {
        'age': randint(18, 65),
        'location': choice(locations),
        'hobbies': choice(hobby_pool,),
        'linkage': randint(50, 200),
        'characteristics': choice(traits),
        'likes_pet_photos': randint(0, 20),
        'clicks_john_share': randint(0, 15),
        'clicks_discount_news': randint(0, 30),
        'clicks_election_news': randint(0, 15),
        'likes_music_videos': randint(0, 25),
        'clicks_food_blog': randint(0, 20),
        'shares_funny_memes': randint(0, 15),
        'clicks_health_articles': randint(0, 15)
    }


def show_content(event=None):
    global step, current_user, prediction, campaign
    text_box.delete(1.0, tk.END)

    if step == 0:
        current_user = generate_random_user()
        prediction = predictor.predict_user_traits(current_user)
        persona = predictor.cluster_to_trait.get(prediction['user_cluster'], 'default')
        user_df = pd.DataFrame([current_user])
        text_box.insert(tk.END, f"\n👤 New Random User:\n{user_df.to_string(index=False)}\n\n🧠 Predicted Persona: {persona}\n")
        step += 1

    elif step == 1:
        default_campaign = predictor.campaign_templates['default'].get(prediction['predicted_campaign_success'], {})
        text_box.insert(tk.END, f"\n📢 Default Campaign:\nMessage: {default_campaign.get('message', 'N/A')}\n"
                          f"Channels: {default_campaign.get('channels')}\n"
                          f"Timing: {default_campaign.get('timing')}\n")
        step += 1

    elif step == 2:
        campaign = predictor.generate_campaign(current_user, discount_percentage=25)
        text_box.insert(tk.END, f"\n🌟 Personalized Campaign for {campaign['personalization_factors']['characteristic']}\n"
                          f"Message: {campaign['message']}\n"
                          f"Channels: {campaign['recommended_channels']}\n"
                          f"Content Type: {campaign['content_type']}\n"
                          f"Timing: {campaign['optimal_timing']}\n"
                          f"Predicted Success Rate: {campaign['predicted_success_rate']:.2%}\n"
                          f"User Segment: {campaign['user_segment']}\n")
        step = 0

# Bind mouse click
root.bind("<Button-1>", show_content)

# Start the UI loop
root.mainloop()

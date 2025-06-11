from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

df= pd.read_csv("Synthetic_User_Data.csv")

X_raw = df[[
    "location", "hobbies", "characteristics", "age", "linkage", "likes_pet_photos",
    "clicks_john_share", "clicks_discount_news", "clicks_election_news",
    "likes_music_videos", "clicks_food_blog", "shares_funny_memes", "clicks_health_articles"
]]
y_raw = df["past_campaign_success"]

# Encode the target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# Split dataset
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)

# Define preprocessing for categorical and multi-label text fields
preprocessor = ColumnTransformer(transformers=[
    ("location", OneHotEncoder(handle_unknown='ignore'), ["location"]),
    ("hobbies", CountVectorizer(tokenizer=lambda x: x.split(', ')), "hobbies"),
    ("characteristics", OneHotEncoder(handle_unknown='ignore'), ["characteristics"])
], remainder='passthrough')  # numerical features passed through

# Define the full pipeline
pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the pipeline
pipeline.fit(X_train_raw, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_test_raw)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

message_bank = {
    ("email_opened", "pet_lover"): "We know your love for furry friends! Share your passion through song 🎤🐾",
    ("email_opened", "political"): "Raise your voice for change. Your vote-worthy voice deserves attention.",
    ("sms_clicked", "funny_memes"): "Your sense of humor is a hit! Let the world see your fun side.",
    ("sms_clicked", "music_lover"): "You vibe to great beats—now let the world vibe to yours!",
    ("social_shared", "influencer"): "Your followers adore you—amplify your voice and go viral!",
    ("social_shared", "trendspotter"): "Always ahead of the curve? Time to trend with your performance.",
    ("sms_clicked", "political"): "Speak up and connect. Important voices deserve to be heard!",
    ("email_opened", "music_lover"): "Your musical vibe is inspiring! Let it shine in your next performance.",
    ("social_shared", "pet_lover"): "You and your pets rock the timeline—let your voice rock it too!",
    # Fallback messages
    ("email_opened", "default"): "Let your voice shine bright—step into the spotlight!",
    ("sms_clicked", "default"): "Your journey starts now! Sing your story.",
    ("social_shared", "default"): "Your talent is unique. Show the world what you’ve got!",
}

def extract_user_theme(user_row):
    if user_row["likes_pet_photos"] > 10:
        return "pet_lover"
    elif user_row["clicks_election_news"] > 15:
        return "political"
    elif user_row["shares_funny_memes"] > 20:
        return "funny_memes"
    elif user_row["likes_music_videos"] > 15:
        return "music_lover"
    elif user_row["linkage"] > 200 or user_row["clicks_john_share"] > 15:
        return "influencer"
    elif user_row["clicks_discount_news"] > 10:
        return "trendspotter"
    else:
        return "default"
def generate_personalized_message(user_row, pipeline, label_encoder):
    # Prepare input features
    input_df = pd.DataFrame([user_row])
    X_input = input_df[[
        "location", "hobbies", "characteristics", "age", "linkage", "likes_pet_photos",
        "clicks_john_share", "clicks_discount_news", "clicks_election_news",
        "likes_music_videos", "clicks_food_blog", "shares_funny_memes", "clicks_health_articles"
    ]]

    # Predict best channel
    pred_encoded = pipeline.predict(X_input)[0]
    predicted_channel = label_encoder.inverse_transform([pred_encoded])[0]

    # Rule-based theme
    theme = extract_user_theme(user_row)

    # Generate personalized message
    message = message_bank.get((predicted_channel, theme),
                               f"Let your talent shine! Join the spotlight via {predicted_channel}."
                               )

    return {
        "user_id": user_row["user_id"],
        "predicted_channel": predicted_channel,
        "theme": theme,
        "message": message
    }

# Example: generate message for first user in your dataset
user_row = df.iloc[0].to_dict()
personalized = generate_personalized_message(user_row, pipeline, label_encoder)

print("=== Personalized Message ===")
print(f"User ID   : {personalized['user_id']}")
print(f"Channel   : {personalized['predicted_channel']}")
print(f"Theme     : {personalized['theme']}")
print(f"Message   : {personalized['message']}")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import warnings

warnings.filterwarnings('ignore')


class UserTraitPredictor:
    """
    A comprehensive ML module for predicting user traits and generating customized campaigns
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = None
        self.target_column = 'past_campaign_success'
        self.trait_clusters = None
        self.campaign_templates = self._initialize_campaign_templates()
        self.cluster_to_trait = {
            0: 'Penny Stockdale',  # analytical
            1: 'Bubba Gunz',  # extrovert
            2: 'Shawn Moonbeam',  # dreamer
            3: 'Brad Cornbread',  # introvert
            4: 'Crypto Brody'  # impulsive
        }

    def _initialize_campaign_templates(self):
        """Initialize campaign templates based on user characteristics and preferences"""
        return {
            'Penny Stockdale': {
                'sms_clicked': {
                    'message': "Be informed: Join our campaign briefing and shape policies that matter to you!",
                    'channels': ['SMS', 'Email'],
                    'content_type': 'policy_news',
                    'timing': 'morning'
                },
                'email_opened': {
                    'message': "Deep Dive: Explore our detailed agenda and policy plans—your insight shapes the future!",
                    'channels': ['Email', 'Push'],
                    'content_type': 'manifesto_details',
                    'timing': 'evening'
                },
                'social_shared': {
                    'message': "Share the facts: Help spread our policy vision for a better future.",
                    'channels': ['Social', 'Email'],
                    'content_type': 'policy_infographics',
                    'timing': 'afternoon'
                }
            },
            'Crypto Brody': {
                'sms_clicked': {
                    'message': "ACT NOW: Show your support—campaign rally starts in 24 hours!",
                    'channels': ['SMS', 'Push'],
                    'content_type': 'rally_alerts',
                    'timing': 'peak_hours'
                },
                'email_opened': {
                    'message': "Last Chance: Pledge your support before midnight!",
                    'channels': ['Email', 'SMS'],
                    'content_type': 'urgency_reminders',
                    'timing': 'evening'
                },
                'social_shared': {
                    'message': "Rally your friends! Your voice can change the election.",
                    'channels': ['Social', 'SMS'],
                    'content_type': 'viral_campaigns',
                    'timing': 'prime_time'
                }
            },
            'Bubba Gunz': {
                'sms_clicked': {
                    'message': " Join our street campaign and meet like-minded changemakers!",
                    'channels': ['SMS', 'Social'],
                    'content_type': 'event_invitations',
                    'timing': 'weekend'
                },
                'email_opened': {
                    'message': "Connect with us: Group actions, volunteer drives, and more!",
                    'channels': ['Email', 'Social'],
                    'content_type': 'community_initiatives',
                    'timing': 'evening'
                },
                'social_shared': {
                    'message': "Gather your friends and mobilize! Campaign events near you.",
                    'channels': ['Social', 'Email'],
                    'content_type': 'group_engagements',
                    'timing': 'weekend'
                }
            },
            'Brad Cornbread': {
                'sms_clicked': {
                    'message': "Your quiet support counts: Sign our petition online from home.",
                    'channels': ['SMS', 'Email'],
                    'content_type': 'online_actions',
                    'timing': 'evening'
                },
                'email_opened': {
                    'message': "Reflect and support: Learn how your beliefs align with our movement.",
                    'channels': ['Email'],
                    'content_type': 'personal_values',
                    'timing': 'quiet_hours'
                },
                'social_shared': {
                    'message': "Support from the sidelines: Promote voter awareness today.",
                    'channels': ['Email', 'App'],
                    'content_type': 'awareness_materials',
                    'timing': 'off_peak'
                }
            },
            'Shawn Moonbeam': {
                'sms_clicked': {
                    'message': "Imagine a better future—support visionary policies with one click!",
                    'channels': ['SMS', 'Email'],
                    'content_type': 'vision_statements',
                    'timing': 'inspiration_hours'
                },
                'email_opened': {
                    'message': "Believe in change: Read our transformation manifesto today.",
                    'channels': ['Email', 'Push'],
                    'content_type': 'hopeful_stories',
                    'timing': 'evening'
                },
                'social_shared': {
                    'message': "Inspire your circle: Share our dream of a just society.",
                    'channels': ['Social', 'Email'],
                    'content_type': 'future_visions',
                    'timing': 'motivational_moments'
                }
            },
            'default': {
                'sms_clicked': {
                    'message': "Support progress: Join our campaign for a fairer future.",
                    'channels': ['SMS'],
                    'content_type': 'general_campaign',
                    'timing': 'standard'
                },
                'email_opened': {
                    'message': "Your voice matters: Back policies that impact real lives.",
                    'channels': ['Email'],
                    'content_type': 'policy_summaries',
                    'timing': 'standard'
                },
                'social_shared': {
                    'message': "Let’s spread the message: Vote for change this election.",
                    'channels': ['Social'],
                    'content_type': 'shareable_missions',
                    'timing': 'standard'
                }
            }
        }
        '''
        return {
            'analytical': {
                'sms_clicked': {
                    'message': "📊 Data-driven insights: {discount}% off tech products. Analyze the savings!",
                    'channels': ['SMS', 'Email'],
                    'content_type': 'tech_news',
                    'timing': 'morning'
                },
                'email_opened': {
                    'message': "Technical Deep Dive: Exclusive {discount}% discount on premium tech gear",
                    'channels': ['Email', 'Push'],
                    'content_type': 'detailed_specs',
                    'timing': 'evening'
                },
                'social_shared': {
                    'message': "Share this exclusive tech deal: {discount}% off innovative gadgets",
                    'channels': ['Social', 'Email'],
                    'content_type': 'infographic',
                    'timing': 'afternoon'
                }
            },
            'impulsive': {
                'sms_clicked': {
                    'message': "⚡ FLASH SALE! {discount}% off - Only 24 hours left!",
                    'channels': ['SMS', 'Push'],
                    'content_type': 'urgent_deals',
                    'timing': 'peak_hours'
                },
                'email_opened': {
                    'message': "Don't miss out! Limited time {discount}% discount expires soon",
                    'channels': ['Email', 'SMS'],
                    'content_type': 'countdown_timer',
                    'timing': 'evening'
                },
                'social_shared': {
                    'message': "Tell your friends! Amazing {discount}% off deal - limited quantity!",
                    'channels': ['Social', 'SMS'],
                    'content_type': 'viral_content',
                    'timing': 'prime_time'
                }
            },
            'extrovert': {
                'sms_clicked': {
                    'message': "🎉 Share the fun! {discount}% off social experiences & events",
                    'channels': ['SMS', 'Social'],
                    'content_type': 'social_activities',
                    'timing': 'weekend'
                },
                'email_opened': {
                    'message': "Connect & Save: {discount}% off group activities and social events",
                    'channels': ['Email', 'Social'],
                    'content_type': 'community_events',
                    'timing': 'evening'
                },
                'social_shared': {
                    'message': "Bring everyone! Group discount {discount}% off social experiences",
                    'channels': ['Social', 'Email'],
                    'content_type': 'group_deals',
                    'timing': 'weekend'
                }
            },
            'introvert': {
                'sms_clicked': {
                    'message': "Personal time deserves personal savings: {discount}% off quiet activities",
                    'channels': ['SMS', 'Email'],
                    'content_type': 'solo_activities',
                    'timing': 'evening'
                },
                'email_opened': {
                    'message': "Your personal sanctuary: {discount}% off home comfort items",
                    'channels': ['Email'],
                    'content_type': 'home_products',
                    'timing': 'quiet_hours'
                },
                'social_shared': {
                    'message': "Quietly amazing deals: {discount}% off personal wellness items",
                    'channels': ['Email', 'App'],
                    'content_type': 'wellness_products',
                    'timing': 'off_peak'
                }
            },
            'dreamer': {
                'sms_clicked': {
                    'message': "✨ Dream big, save bigger: {discount}% off aspirational products",
                    'channels': ['SMS', 'Email'],
                    'content_type': 'lifestyle_products',
                    'timing': 'inspiration_hours'
                },
                'email_opened': {
                    'message': "Turn dreams into reality: {discount}% off life-changing products",
                    'channels': ['Email', 'Push'],
                    'content_type': 'transformation_stories',
                    'timing': 'evening'
                },
                'social_shared': {
                    'message': "Inspire others: Share this {discount}% discount on dream products",
                    'channels': ['Social', 'Email'],
                    'content_type': 'inspirational_content',
                    'timing': 'motivational_moments'
                }
            },
            'default': {
                'sms_clicked': {
                    'message': "Great savings await: {discount}% off selected items",
                    'channels': ['SMS'],
                    'content_type': 'general_offers',
                    'timing': 'standard'
                },
                'email_opened': {
                    'message': "Exclusive offer: {discount}% discount just for you",
                    'channels': ['Email'],
                    'content_type': 'personalized_offers',
                    'timing': 'standard'
                },
                'social_shared': {
                    'message': "Share the savings: {discount}% off great products",
                    'channels': ['Social'],
                    'content_type': 'shareable_deals',
                    'timing': 'standard'
                }
            }
        }
        '''

    def preprocess_data(self, df, is_training=True):
        """Preprocess the data for training or prediction"""
        # Create feature columns
        behavioral_features = [
            'likes_pet_photos', 'clicks_john_share', 'clicks_discount_news',
            'clicks_election_news', 'likes_music_videos', 'clicks_food_blog',
            'shares_funny_memes', 'clicks_health_articles'
        ]

        demographic_features = ['age', 'linkage']
        categorical_features = ['location', 'hobbies', 'characteristics']

        # Encode categorical variables
        df_processed = df.copy()

        # Handle hobbies (multi-label)
        hobby_categories = ['tech', 'biking', 'memes', 'gaming', 'pets', 'cooking',
                            'tiktok', 'music', 'travel', 'reading', 'hiking', 'dancing',
                            'singing', 'fashion', 'gardening']

        for hobby in hobby_categories:
            df_processed[f'hobby_{hobby}'] = df_processed['hobbies'].str.contains(hobby, na=False).astype(int)

        if is_training:
            # Fit and transform encoders for training data
            location_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            location_encoded = location_encoder.fit_transform(df_processed[['location']])
            location_feature_names = [f'location_{loc}' for loc in location_encoder.categories_[0]]

            char_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            char_encoded = char_encoder.fit_transform(df_processed[['characteristics']])
            char_feature_names = [f'char_{char}' for char in char_encoder.categories_[0]]

            # Store encoders
            self.encoders['location'] = location_encoder
            self.encoders['characteristics'] = char_encoder

            # Store feature column names
            self.feature_columns = (demographic_features + behavioral_features +
                                    location_feature_names + char_feature_names +
                                    [f'hobby_{hobby}' for hobby in hobby_categories])
        else:
            # Use existing encoders for prediction data
            location_encoded = self.encoders['location'].transform(df_processed[['location']])
            char_encoded = self.encoders['characteristics'].transform(df_processed[['characteristics']])

        # Create final feature matrix
        feature_data = np.concatenate([
            df_processed[demographic_features + behavioral_features].values,
            location_encoded,
            char_encoded,
            df_processed[[f'hobby_{hobby}' for hobby in hobby_categories]].values
        ], axis=1)

        if is_training and self.target_column in df_processed.columns:
            return feature_data, df_processed[self.target_column]
        else:
            return feature_data, None

    def train_models(self, df):
        """Train multiple models for comparison"""
        X, y = self.preprocess_data(df, is_training=True)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler

        # Define models to train
        models_to_train = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }

        results = {}

        for name, model in models_to_train.items():
            print(f"Training {name}...")

            if name == 'logistic_regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'report': classification_report(y_test, y_pred)
            }

            print(f"{name} Accuracy: {accuracy:.4f}")
            print(f"Classification Report:\n{results[name]['report']}\n")

        # Store the best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        self.models['best_classifier'] = results[best_model_name]['model']
        self.models['best_model_name'] = best_model_name

        print(f"Best model: {best_model_name} with accuracy: {results[best_model_name]['accuracy']:.4f}")

        # Train clustering model for user segmentation
        self.train_clustering(X_train_scaled)

        return results

    def train_clustering(self, X_scaled):
        """Train clustering model for user segmentation"""
        # Determine optimal number of clusters using elbow method
        inertias = []
        k_range = range(2, 11)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)

        # Use k=5 as a reasonable default (you can adjust based on elbow method)
        optimal_k = 5
        self.trait_clusters = KMeans(n_clusters=optimal_k, random_state=42)
        self.trait_clusters.fit(X_scaled)

        print(f"Clustering completed with {optimal_k} clusters")

    def predict_user_traits(self, user_data):
        """Predict user traits and cluster membership"""
        if isinstance(user_data, dict):
            user_df = pd.DataFrame([user_data])
        else:
            user_df = user_data.copy()

        # Preprocess user data (not training, so no target column expected)
        X_user, _ = self.preprocess_data(user_df, is_training=False)

        # Scale features
        X_user_scaled = self.scalers['standard'].transform(X_user)

        # Predict campaign success probability
        if self.models['best_model_name'] == 'logistic_regression':
            campaign_prediction = self.models['best_classifier'].predict(X_user_scaled)
            campaign_proba = self.models['best_classifier'].predict_proba(X_user_scaled)
        else:
            campaign_prediction = self.models['best_classifier'].predict(X_user)
            campaign_proba = self.models['best_classifier'].predict_proba(X_user)

        # Predict cluster membership
        cluster_prediction = self.trait_clusters.predict(X_user_scaled)

        results = []
        for i in range(len(user_df)):
            user_result = {
                'predicted_campaign_success': campaign_prediction[i],
                'success_probability': dict(zip(
                    self.models['best_classifier'].classes_,
                    campaign_proba[i]
                )),
                'user_cluster': cluster_prediction[i],
                'characteristics': user_df.iloc[i][
                    'characteristics'] if 'characteristics' in user_df.columns else 'unknown'
            }
            results.append(user_result)

        return results if len(results) > 1 else results[0]

    def generate_campaign(self, user_data, discount_percentage=20):
        """Generate personalized campaign based on user traits"""
        # Get user predictions
        prediction = self.predict_user_traits(user_data)

        # Extract user characteristics
        user_char = self.cluster_to_trait.get(prediction['user_cluster'], 'default')
        print (f"Campaign generated for {user_char}")
        predicted_success_type = prediction['predicted_campaign_success']
        success_probabilities = prediction['success_probability']

        # Get campaign template
        if user_char in self.campaign_templates:
            char_templates = self.campaign_templates[user_char]
        else:
            char_templates = self.campaign_templates['default']

        # Select template based on predicted success type
        if predicted_success_type in char_templates:
            template = char_templates[predicted_success_type]
        else:
            # Fallback to most likely success type
            most_likely_type = max(success_probabilities.keys(),
                                   key=lambda k: success_probabilities[k])
            template = char_templates.get(most_likely_type, char_templates['email_opened'])

        # Generate personalized campaign
        campaign = {
            'message': template['message'].format(discount=discount_percentage),
            'recommended_channels': template['channels'],
            'content_type': template['content_type'],
            'optimal_timing': template['timing'],
            'predicted_success_rate': max(success_probabilities.values()),
            'user_segment': f"Cluster_{prediction['user_cluster']}",
            'personalization_factors': {
                'characteristic': user_char,
                'predicted_response': predicted_success_type,
                'cluster': prediction['user_cluster']
            }
        }

        return campaign

    def save_model(self, filepath):
        """Save the trained model and preprocessors"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_columns': self.feature_columns,
            'trait_clusters': self.trait_clusters,
            'campaign_templates': self.campaign_templates
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a previously trained model"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.encoders = model_data['encoders']
        self.feature_columns = model_data['feature_columns']
        self.trait_clusters = model_data['trait_clusters']
        self.campaign_templates = model_data['campaign_templates']
        print(f"Model loaded from {filepath}")


# Example usage and demonstration
def main():
    # Load the data
    df = pd.read_csv('Synthetic_User_Data.csv')

    # Initialize and train the model
    predictor = UserTraitPredictor()
    results = predictor.train_models(df)

    # Example: Predict for a new user
    new_user = {
        'age': 20,
        'location': 'Oslo',
        'hobbies': 'music',
        'linkage': 150,
        'characteristics': 'extrovert',
        'likes_pet_photos': 10,
        'clicks_john_share': 5,
        'clicks_discount_news': 15,
        'clicks_election_news': 8,
        'likes_music_videos': 20,
        'clicks_food_blog': 12,
        'shares_funny_memes': 8,
        'clicks_health_articles': 10
    }

    # Generate personalized campaign
    campaign = predictor.generate_campaign(new_user, discount_percentage=25)

    print("Generated Campaign:")
    print(f"Message: {campaign['message']}")
    print(f"Channels: {campaign['recommended_channels']}")
    print(f"Content Type: {campaign['content_type']}")
    print(f"Timing: {campaign['optimal_timing']}")
    print(f"Predicted Success Rate: {campaign['predicted_success_rate']:.2%}")
    print(f"User Segment: {campaign['user_segment']}")

    # Save the model
    predictor.save_model('user_trait_model.pkl')

    return predictor


if __name__ == "__main__":
    predictor = main()
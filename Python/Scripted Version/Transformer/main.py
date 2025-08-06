#====================================================================================================================
# Imports and Constants
#====================================================================================================================
#%%

# Core Libraries
import re
import numpy as np
import pandas as pd
import traceback

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import layers, Model

# Preprocessing & Dimensionality Reduction
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

# Visualization
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import matplotlib.patches as mpatches

# Machine Learning Models
import xgboost as xgb
    

# Constants
RANDOM_SEED = 42
TEST_SIZE = 0.2
MAX_TOKENS = 1000
OUTPUT_SEQUENCE_LENGTH = 50
EMBED_DIM = 32
NUM_HEADS = 2
NUM_TRANSFORMER_BLOCKS = 2

# Parameter for Job Description Simulation
FRAC = 1

#====================================================================================================================
# Data Preprocessing Module
#====================================================================================================================
class TabularDataPreprocessor:
    @staticmethod
    def load_and_prepare_data(filepath):
        """Load and prepare the dataset with job descriptions"""
        df = pd.read_csv(filepath)
        
        # Generate job descriptions
        risky_descriptions = [
            # 1. Extended temporary work with hidden instability
            """Demonstrated exceptional adaptability through successive short-term contracts (3-6 months duration) at multiple luxury hotel properties, 
            consistently brought in to stabilize operations during periods of critical staffing shortages and unexpected leadership transitions. 
            Maintained 85-90% guest satisfaction metrics despite inheriting teams with 40-50% vacancy rates and outdated property management systems, 
            often working 60+ hour weeks to cover multiple roles during transitional phases.""",

            # 2. Financial distress with positive framing
            """Spearheaded comprehensive cost restructuring initiatives at a 300-room urban hotel during Chapter 11 proceedings, 
            innovating vendor payment solutions that extended terms from net-30 to net-90 while maintaining 70% service levels. 
            Reduced operational expenses by 35% through strategic workforce optimization and deferred capital expenditures, 
            all while navigating complex creditor negotiations and preserving essential guest amenities.""",

            # 3. Extreme turnover masked as training achievement
            """Revolutionized housekeeping operations at a high-turnover resort property by developing accelerated training protocols 
            that enabled new staff to achieve 80% productivity within 48 hours of hiring. Successfully onboarded and managed 
            12 distinct housekeeping teams over an 8-month period (150% annualized turnover rate), maintaining consistent cleanliness 
            scores through intensive daily quality control inspections and performance coaching.""",

            # 4. Ownership chaos with achievement language
            """Led front office operations through three separate ownership transitions in an 18-month period, 
            implementing temporary solutions that bridged incompatible reservation systems and maintained 75% occupancy levels. 
            Developed comprehensive operational playbooks for each transition that reduced typical 4-week stabilization periods to 10 days, 
            despite working with management teams that changed completely with each ownership transfer.""",

            # 5. Gig economy instability
            """Provided premium hospitality services through an exclusive staffing collective, completing 14 short-term assignments 
            (2-4 month durations) at various luxury properties experiencing sudden management gaps. Specialized in operational 
            triage scenarios requiring immediate adaptation to unfamiliar property management systems and team dynamics, 
            often achieving 30% productivity improvements within first 72 hours of engagement.""",

            # 6. Systemic understaffing
            """Single-handedly managed all guest services for a 250-room boutique hotel during prolonged staffing crisis, 
            covering front desk, concierge, and night audit responsibilities across 14-hour daily shifts. 
            Implemented emergency protocols that reduced check-in times from 8 minutes to 3 minutes despite working 
            with manual backup systems during 6-week property management system outage.""",

            # 7. Seasonal volatility
            """Pioneered innovative staffing model for seasonal mountain resort that dynamically scaled teams from 
            45 full-time employees in peak season down to 8 core staff during off-months, while maintaining 
            70% guest satisfaction scores through strategic service reductions and automated solutions. 
            Managed complex logistics of housing and transporting temporary international staff on J-1 visas.""",

            # 8. Hidden layoffs
            """Orchestrated seamless operational transition during corporate-mandated 40% headcount reduction, 
            redesigning workflows to maintain 65% productivity levels with remaining staff. Implemented 
            knowledge transfer protocols that captured institutional expertise from departing employees, 
            while establishing cross-training programs to mitigate single-point-of-failure risks.""",

            # 9. Chronic payment issues
            """Transformed accounts payable processes during prolonged cash flow constraints, developing 
            innovative vendor payment schedules that prioritized critical suppliers while extending terms 
            for others by 60-90 days. Negotiated 15% discounts with key partners in exchange for deferred 
            payments, preventing any service disruptions despite 45-day average payment delays.""",

            # 10. Management churn
            """Steered operations through 5 different general managers in 11-month period, 
            maintaining consistent service standards despite radically shifting strategic priorities 
            with each leadership change. Became de facto institutional knowledge repository, 
            training each new management team on property-specific operational nuances.""",

            # 11. Technology failures
            """Bridged critical technology gaps during botched property management system migration, 
            developing manual workarounds that maintained 70% reservation accuracy despite 
            system synchronization failures. Trained entire staff on backup procedures that 
            prevented $250k in potential lost revenue during 3-week stabilization period.""",

            # 12. Regulatory non-compliance
            """Maintained uninterrupted operations during 6-month licensing review period, 
            implementing interim compliance measures that addressed 85% of regulatory 
            requirements while final approvals were pending. Navigated complex inspection 
            protocols through careful documentation and daily operational adjustments.""",

            # 13. Benefit cuts
            """Restructured employee compensation package during financial reorganization, 
            replacing traditional benefits with flexible scheduling options and performance bonuses. 
            Maintained 70% staff retention through transparent communication and phased implementation, 
            despite 30% reduction in total compensation costs.""",

            # 14. Safety compromises
            """Balanced stringent budget constraints with safety compliance requirements, 
            achieving 80% inspection scores through creative maintenance scheduling and 
            selective equipment upgrades. Implemented temporary safety protocols that 
            met minimum standards while deferring $150k in capital improvements.""",

            # 15. Vendor instability
            """Navigated frequent vendor turnover during supply chain disruptions, 
            qualifying 12 new suppliers in 6-month period while maintaining quality standards. 
            Negotiated emergency 30-day payment terms with all new vendors to preserve 
            cash flow during revenue shortfalls.""",

            # 16. Labor disputes
            """Maintained operations during 3-month labor negotiations, covering 
            multiple front-line positions while training temporary replacements. 
            Implemented contingency plans that preserved 75% of normal service 
            levels despite 40% reduction in available staff.""",

            # 17. Disaster recovery
            """Led operational recovery after catastrophic flood damage, 
            rebuilding key systems with 60% reduced budget and temporary facilities. 
            Maintained 50% occupancy using creative room configurations and 
            heavily modified service offerings during 8-month renovation.""",

            # 18. Brand transition
            """Managed complex rebranding from independent hotel to franchise flag, 
            reconciling incompatible operating standards during 6-month transition. 
            Trained staff on entirely new service protocols while maintaining 
            70% guest satisfaction scores throughout disruptive changes.""",

            # 19. Pandemic response
            """Architected COVID-19 operational model that allowed 50% occupancy 
            despite 70% staff reductions and stringent safety protocols. Developed 
            cross-functional teams capable of covering 3-4 positions each, while 
            implementing contactless technologies to maintain service standards.""",

            # 20. Chronic maintenance
            """Preserved aging physical plant through creative maintenance solutions, 
            deferring $500k in capital expenditures while maintaining 80% guest 
            satisfaction. Implemented intensive daily inspection routines that 
            identified and addressed issues before they impacted operations."""
        ]


        generic_descriptions = [
            # 1. Institutional stability
            """Steadily progressed through ranks over 8-year tenure at flagship Marriott property, 
            from front desk associate to rooms division manager overseeing 120 employees. 
            Maintained consistent 96%+ guest satisfaction scores and perfect quality audit results 
            while developing 15 team members into management positions.""",

            # 2. System excellence
            """Led enterprise-wide property management system upgrade across 5-hotel portfolio, 
            coordinating 18-month migration that achieved 99.9% data integrity with zero 
            operational downtime. Developed comprehensive training programs adopted as 
            corporate standard for all future technology implementations.""",

            # 3. Process mastery
            """Transformed housekeeping operations through RFID inventory tracking system, 
            achieving 98% inspection consistency across 200+ rooms daily for 12 consecutive 
            quarters. Reduced linen loss by 40% and cleaning supply costs by 25% while 
            improving room readiness times by 30 minutes on average.""",

            # 4. Staff development
            """Architected employee development program that increased retention rates from 
            75% to 92% over 3-year period. Implemented competency-based promotion system 
            that reduced management training time by 50% while improving department 
            performance metrics by 15% across all categories.""",

            # 5. Revenue optimization
            """Pioneered dynamic pricing strategy that increased RevPAR by 22% without 
            compromising occupancy rates. Integrated market analytics with operational 
            capacity planning to optimize rate structures across all room categories, 
            generating $1.2M in incremental annual revenue.""",

            # 6. Quality leadership
            """Directed quality assurance program that earned AAA Five Diamond rating 
            for 4 consecutive years - the longest streak in property's 30-year history. 
            Established inspection protocols that identified and corrected 95% of 
            service gaps before they impacted guest experiences.""",

            # 7. Renovation expertise
            """Managed $4.5M guest room renovation completed 2 weeks ahead of schedule 
            and 8% under budget. Developed phased implementation plan that allowed 
            80% occupancy throughout construction while maintaining 94% satisfaction scores.""",

            # 8. Safety excellence
            """Maintained perfect safety record for 7+ years through proactive equipment 
            maintenance and comprehensive training programs. Reduced workers compensation 
            claims by 60% through ergonomic assessments and preventive safety protocols.""",

            # 9. Group sales
            """Built corporate accounts program from scratch that grew to $3.2M in annual 
            group business revenue. Cultivated relationships with 25+ Fortune 500 companies, 
            maintaining 92% repeat business rate through customized service agreements.""",

            # 10. Technology innovation
            """Championed mobile technology integration that increased digital check-ins to 
            75% of arrivals within 18 months. Reduced front desk staffing requirements by 
            30% while improving guest satisfaction scores through personalized mobile 
            concierge services.""",

            # 11. Sustainability
            """Led environmental initiatives that reduced energy consumption by 35% and 
            water usage by 28% while maintaining luxury service standards. Achieved 
            prestigious LEED Gold certification through comprehensive operational changes 
            and staff engagement programs.""",

            # 12. Community engagement
            """Forged partnerships with 12 local cultural institutions that increased 
            package sales by 40% and boosted TripAdvisor ranking from #42 to #8 in 
            market. Developed signature experiences that became destination highlights 
            for 25% of guests.""",

            # 13. Loyalty growth
            """Transformed loyalty program from 15,000 to 85,000 active members in 
            3 years through targeted marketing and tiered benefits. Increased member 
            repeat stays from 1.2 to 2.8 annually, driving 30% of total room revenue.""",

            # 14. Culinary excellence
            """Elevated restaurant from 3.5 to 4.5-star average on review platforms 
            through menu innovation and service enhancements. Increased covers by 40% 
            while improving food cost percentage by 5 points through strategic sourcing.""",

            # 15. Revenue synergy
            """Chaired cross-departmental revenue strategy team that aligned sales, 
            marketing and operations to exceed annual targets by 12-15% for 3 
            consecutive years. Developed predictive modeling that improved group 
            booking pacing by 25%.""",

            # 16. Training innovation
            """Created immersive training academy that reduced new hire ramp time from 
            6 weeks to 10 days while improving service quality metrics by 18%. 
            Program became corporate benchmark adopted across 12 other properties.""",

            # 17. Brand standards
            """Maintained 98% compliance with stringent brand standards across 450 
            measurable items for 5+ years. Developed self-audit system that reduced 
            corporate inspection preparation time by 75% while improving scores.""",

            # 18. Crisis preparedness
            """Designed emergency response protocols adopted as regional best practices 
            after successfully managing through hurricane and wildfire events. System 
            reduced typical recovery time from 72 to 24 hours for severe disruptions.""",

            # 19. Asset management
            """Oversaw $15M capital improvement plan that extended property lifecycle 
            by 10+ years while modernizing all guest-facing areas. Phased construction 
            maintained 85% occupancy throughout 18-month project.""",

            # 20. Market leadership
            """Elevated property to #1 ranking in market through comprehensive service 
            enhancements and strategic pricing. Grew ADR by 28% over 3 years while 
            improving satisfaction scores from 88% to 96% through staff empowerment 
            initiatives."""
        ]


        
        # Create risky job descriptions for high-risk cases
        risky_pool = df[(df['Default'] == 1) & (df['HasDependents'] == 'Yes')]
        risky_sample = risky_pool.sample(frac=FRAC, random_state=RANDOM_SEED)
        
        df['JobDescription'] = None
        for i, idx in enumerate(risky_sample.index):
            df.at[idx, 'JobDescription'] = risky_descriptions[i % len(risky_descriptions)]
        
        # Fill remaining with generic descriptions
        remaining_indices = df[df['JobDescription'].isna()].index
        df.loc[remaining_indices, 'JobDescription'] = np.random.choice(
            generic_descriptions, size=len(remaining_indices))
        
        return df
    
    @staticmethod
    def preprocess_data(df):
        """Split data and prepare features"""
        # Split data
        train_data, test_data = train_test_split(
            df, test_size=TEST_SIZE, random_state=RANDOM_SEED)
        
        # Save raw text before encoding
        train_data_raw = train_data.copy()
        test_data_raw = test_data.copy()
        
        # Separate labels
        y_train = train_data['Default']
        y_test = test_data['Default']
        
        # Define columns
        categorical_columns = ['Education', 'EmploymentType', 'MaritalStatus',
                             'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
        
        numerical_columns = train_data.select_dtypes(
            include=['int64', 'float64']).columns.tolist()
        numerical_columns = [col for col in numerical_columns 
                            if col not in categorical_columns + ['Default']]
        
        # Encode categorical columns
        for col in categorical_columns:
            train_data[col] = train_data[col].astype('category').cat.codes
            test_data[col] = test_data[col].astype('category').cat.codes
        
        # Get cardinalities
        cat_cardinalities = [train_data[col].nunique() for col in categorical_columns]
        cat_features_info = [(card, EMBED_DIM) for card in cat_cardinalities]
        
        # Scale numerical features
        scaler = StandardScaler()
        train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])
        test_data[numerical_columns] = scaler.transform(test_data[numerical_columns])
        
        # Prepare text vectorizer
        text_vectorizer = layers.TextVectorization(
            max_tokens=MAX_TOKENS,
            output_mode='int',
            output_sequence_length=OUTPUT_SEQUENCE_LENGTH
        )
        text_vectorizer.adapt(train_data_raw['JobDescription'].fillna('').astype(str).values)
        
        vocab = text_vectorizer.get_vocabulary()

        print(vocab)

        # Vectorize text
        X_train_text = text_vectorizer(train_data_raw['JobDescription'].fillna('').astype(str).values)
        X_test_text = text_vectorizer(test_data_raw['JobDescription'].fillna('').astype(str).values)
        
        # Convert to tensors
        X_train = {
            'categorical': tf.convert_to_tensor(train_data[categorical_columns].values), 
            'numerical': tf.convert_to_tensor(train_data[numerical_columns].values),
            'text': X_train_text
        }
        
        X_test = {
            'categorical': tf.convert_to_tensor(test_data[categorical_columns].values),
            'numerical': tf.convert_to_tensor(test_data[numerical_columns].values),
            'text': X_test_text
        }
        
        y_train = tf.convert_to_tensor(y_train.values)
        y_test = tf.convert_to_tensor(y_test.values)
        
        return X_train, X_test, y_train, y_test, cat_features_info, len(numerical_columns)

#====================================================================================================================
# Model Building Module
#====================================================================================================================
class TransformerBlock(layers.Layer):
    """Improved Transformer block with better compatibility"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(embed_dim * 2, activation="gelu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)

class TabTransformerModel:
    """Compatible TabTransformer with improvements"""
    @staticmethod
    def build_model(cat_features_info, num_numerical):
        """Build the full TabTransformer model"""
        # Input layers (unchanged)
        categorical_inputs = layers.Input(shape=(len(cat_features_info),), name='categorical_inputs')
        numerical_inputs = layers.Input(shape=(num_numerical,), name='numerical_inputs')
        text_inputs = layers.Input(shape=(OUTPUT_SEQUENCE_LENGTH,), name='text_inputs')
        
        # Categorical processing (with improved embeddings)
        embedded_cats = []
        for i, (card, dim) in enumerate(cat_features_info):
            emb = layers.Embedding(
                input_dim=card, 
                output_dim=dim,
                embeddings_regularizer=tf.keras.regularizers.l2(1e-5)
            )(categorical_inputs[:, i:i+1])
            embedded_cats.append(emb)
        
        x_cat = layers.Concatenate(axis=1)(embedded_cats)
        
        # Transformer blocks (improved)
        for _ in range(NUM_TRANSFORMER_BLOCKS):
            x_cat = TransformerBlock(EMBED_DIM, NUM_HEADS)(x_cat)
        
        x_cat = layers.Flatten()(x_cat)
        
        # Numerical features (with batch norm)
        x_num = layers.BatchNormalization()(numerical_inputs)
        x_num = layers.Dense(32, activation='gelu')(x_num)
        
        # Text processing (improved)
        x_text = layers.Embedding(
            input_dim=MAX_TOKENS, 
            output_dim=32,
            mask_zero=True
        )(text_inputs)
        x_text = TransformerBlock(32, 2)(x_text)
        x_text = layers.GlobalAveragePooling1D()(x_text)
        
        # Feature combination (simplified but effective)
        x = layers.Concatenate()([x_cat, x_num, x_text])
        x = layers.BatchNormalization()(x)
        x = layers.Dense(64, activation='gelu')(x)
        x = layers.Dense(32, activation='gelu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = Model(
            inputs=[categorical_inputs, numerical_inputs, text_inputs],
            outputs=outputs
        )
        
        return model
    
#====================================================================================================================
# Training and Evaluation Module
#====================================================================================================================
class ModelTrainer:
    @staticmethod
    def train_model(model, X_train, y_train):
        """Train the TabTransformer model"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
        ]
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        history = model.fit(
            (X_train['categorical'], X_train['numerical'], X_train['text']),
            y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks
        )
        
        return model, history

class ModelEvaluator:
    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """Evaluate the model and print metrics"""
        y_pred_proba = model.predict((X_test['categorical'], X_test['numerical'], X_test['text']))
        y_pred = (y_pred_proba.flatten() > 0.5).astype(int)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Print results
        print("Evaluation Results:")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"AUC:       {auc:.4f}")
        
        # Get and print confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:\n", cm)
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        
        # Plot styled confusion matrix
        cmap = sns.cubehelix_palette(start=0.5, rot=-0.5, dark=0.3, light=0.85, as_cmap=True)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap=cmap,
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1']
        )
        plt.title('Model Confusion Matrix')
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()
        
        return acc, auc, f1
    
    @staticmethod
    def evaluate_xgboost(model, X_train, y_train, X_test, y_test, use_embeddings=True):
        """Fixed XGBoost evaluation with proper one-hot encoding for categoricals when not using embeddings"""
        if use_embeddings:
            # Create input mapping from our data structure to model's expected names
            train_inputs = {
                'categorical_inputs': X_train['categorical'],
                'numerical_inputs': X_train['numerical'],
                'text_inputs': X_train['text']
            }
            
            test_inputs = {
                'categorical_inputs': X_test['categorical'],
                'numerical_inputs': X_test['numerical'],
                'text_inputs': X_test['text']
            }

            # Create embedding model
            embedding_model = Model(
                inputs=model.inputs,
                outputs=model.layers[-2].output
            )
            
            # Get embeddings using properly mapped inputs
            X_train_emb = embedding_model.predict(train_inputs)
            X_test_emb = embedding_model.predict(test_inputs)
            
            features = X_train_emb
            test_features = X_test_emb
            title = "XGBoost on Transformer Embeddings"
        else:
            # Use raw features with proper one-hot encoding for categoricals
            categorical_columns = ['Education', 'EmploymentType', 'MaritalStatus',
                                'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
            
            # Convert back to pandas DataFrames for one-hot encoding
            train_cat_df = pd.DataFrame(X_train['categorical'].numpy(), columns=categorical_columns)
            test_cat_df = pd.DataFrame(X_test['categorical'].numpy(), columns=categorical_columns)
            
            # One-hot encode categoricals
            train_cat_encoded = pd.get_dummies(train_cat_df, columns=categorical_columns)
            test_cat_encoded = pd.get_dummies(test_cat_df, columns=categorical_columns)
            
            # Ensure test data has same columns as train (in case some categories are missing)
            missing_cols = set(train_cat_encoded.columns) - set(test_cat_encoded.columns)
            for col in missing_cols:
                test_cat_encoded[col] = 0
            test_cat_encoded = test_cat_encoded[train_cat_encoded.columns]
            
            # Combine with numerical features
            train_num_df = pd.DataFrame(X_train['numerical'].numpy())
            test_num_df = pd.DataFrame(X_test['numerical'].numpy())
            
            X_train_combined = pd.concat([train_cat_encoded, train_num_df], axis=1)
            X_test_combined = pd.concat([test_cat_encoded, test_num_df], axis=1)
            
            features = X_train_combined.values
            test_features = X_test_combined.values
            title = "XGBoost on Raw Features (One-Hot Encoded)"
        
        # Train and evaluate XGBoost
        xgb_clf = xgb.XGBClassifier(use_label_encoder=False, 
                                eval_metric='logloss', 
                                random_state=RANDOM_SEED)
        xgb_clf.fit(features, y_train.numpy())
        
        y_pred = xgb_clf.predict(test_features)
        acc = accuracy_score(y_test.numpy(), y_pred)
        cm = confusion_matrix(y_test.numpy(), y_pred)
        
        ModelEvaluator.plot_confusion_matrix(cm, f"XGBoost Confusion Matrix with Embeddings: {use_embeddings}")
        
        # Compute predicted probabilities for positive class
        y_probs = xgb_clf.predict_proba(test_features)[:, 1]

        # Compute AUC score
        auc = roc_auc_score(y_test.numpy(), y_probs)
        print(f"XGBoost AUC: {auc:.4f}")

        print("\nClassification Report:\n", classification_report(y_test, y_pred))

        print(f"{title} Accuracy: {acc:.4f}")
        return acc
    
    @staticmethod
    def plot_confusion_matrix(cm, title):
        cmap = sns.cubehelix_palette(start=0.5, rot=-0.5, dark=0.3, light=0.85, as_cmap=True)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap=cmap,
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1']
        )
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()

#====================================================================================================================
# MLP Model Module (Tensorflow Version)
#====================================================================================================================
class MLPModel:
    @staticmethod
    def build_model(input_dim, hidden_dim=64, output_dim=1):
        """Build a simple MLP model using TensorFlow/Keras"""
        inputs = layers.Input(shape=(input_dim,))
        
        x = layers.Dense(hidden_dim, activation='relu')(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(hidden_dim, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(output_dim, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model

class MLPTrainer:
    @staticmethod
    def train_and_evaluate(X_train, y_train, X_test, y_test, epochs=100):
        """Train and evaluate MLP on raw features (no embeddings)"""

        # 1. Prepare features
        X_train_combined = np.concatenate([
            X_train['categorical'].numpy(),  # Label-encoded categoricals
            X_train['numerical'].numpy()     # Pre-scaled numerical features
        ], axis=1)
        
        X_test_combined = np.concatenate([
            X_test['categorical'].numpy(),
            X_test['numerical'].numpy()
        ], axis=1)

        # 2. Build model
        input_dim = X_train_combined.shape[1]
        model = MLPModel.build_model(input_dim)
        
        # 3. Compile and train
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        history = model.fit(
            X_train_combined, y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ],
            verbose=1
        )

        # 4. Evaluate
        y_pred_proba = model.predict(X_test_combined)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        acc = accuracy_score(y_test.numpy(), y_pred)
        cm = confusion_matrix(y_test.numpy(), y_pred)

        # Compute predicted probabilities for positive class
        y_probs = model.predict(X_test_combined).ravel()

        # Compute AUC score
        auc = roc_auc_score(y_test, y_probs)
        print(f"XGBoost AUC: {auc:.4f}")

        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print(f'MLP (Raw Features) Accuracy: {acc:.4f}')
        ModelEvaluator.plot_confusion_matrix(cm, "MLP Confusion Matrix (Raw Features)")
        
        return acc, cm


#====================================================================================================================
# Main Execution
#====================================================================================================================
if __name__ == "__main__":
    # 1. Data Preprocessing
    preprocessor = TabularDataPreprocessor()
    df = preprocessor.load_and_prepare_data("data/df_small_sampled.csv")
    X_train, X_test, y_train, y_test, cat_features_info, num_numerical = preprocessor.preprocess_data(df)
    
    # 2. Model Building
    model_builder = TabTransformerModel()
    model = model_builder.build_model(cat_features_info, num_numerical)
    
    # 3. Model Training
    trainer = ModelTrainer()
    trained_model, history = trainer.train_model(model, X_train, y_train)
    
    # 4. Model Evaluation
    evaluator = ModelEvaluator()
    evaluator.evaluate_model(trained_model, X_test, y_test)
    
    # 5. XGBoost Evaluation
    print("\nEvaluating XGBoost on different feature sets:")
    evaluator.evaluate_xgboost(trained_model, X_train, y_train, X_test, y_test, use_embeddings=True)
    evaluator.evaluate_xgboost(trained_model, X_train, y_train, X_test, y_test, use_embeddings=False)

    # 6. MLP Evaluation
    print("\nEvaluating MLP on raw features:")
    mlp_acc, mlp_cm = MLPTrainer.train_and_evaluate(X_train, y_train, X_test, y_test)

    # 7. Visualization of Embeddings
    # Extract embeddings for test data using the embedding model
    embedding_model = Model(
        inputs=model.inputs,
        outputs=model.layers[-3].output  # The layer before final Dense layers
    )

    # Predict embeddings for test set
    test_embeddings = embedding_model.predict((X_test['categorical'], X_test['numerical'], X_test['text']))

    # Choose dimensionality reduction method: t-SNE or PCA
    def plot_embeddings(embeddings, labels, method='tsne'):
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=RANDOM_SEED)
            reduced_emb = reducer.fit_transform(embeddings)
        elif method == 'pca':
            reducer = PCA(n_components=2)
            reduced_emb = reducer.fit_transform(embeddings)
        else:
            raise ValueError("Method must be 'tsne' or 'pca'")

        cubehelix_colors = sns.cubehelix_palette(
            start=0.5, rot=-0.5,
            dark=0.7, light=0.3,
            n_colors=2,
            reverse=False
        )
        label_to_color = {0: cubehelix_colors[0], 1: cubehelix_colors[1]}
        point_colors = [label_to_color[label] for label in labels]

        plt.figure(figsize=(8, 6))
        plt.scatter(
            reduced_emb[:, 0], reduced_emb[:, 1],
            c=point_colors, alpha=0.7, edgecolor='none'
        )
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')

        legend_elements = [
            mpatches.Patch(facecolor=cubehelix_colors[0], label='Class 0'),
            mpatches.Patch(facecolor=cubehelix_colors[1], label='Class 1')
        ]
        plt.legend(handles=legend_elements, title='Default', loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Plot embeddings with true labels
    plot_embeddings(test_embeddings, y_test.numpy(), method='tsne')


    # 8. Visualize Centroid with PCA
    classes = np.unique(y_test)
    centroids = np.array([test_embeddings[y_test == c].mean(axis=0) for c in classes])

    # Reduce centroids to 2D
    classes = np.unique(y_test)
    centroids = np.array([test_embeddings[y_test == c].mean(axis=0) for c in classes])

    # Fit PCA on all embeddings and transform both embeddings and centroids
    pca = PCA(n_components=2)
    pca.fit(test_embeddings)
    embeddings_2d = pca.transform(test_embeddings)
    centroids_2d = pca.transform(centroids)

    # Define cubehelix colors consistent with your palette
    cubehelix_colors = sns.cubehelix_palette(
        start=0.5, rot=-0.5,
        dark=0.7, light=0.3,
        n_colors=2,
        reverse=False
    )

    y_test_np = y_test.numpy() if hasattr(y_test, 'numpy') else np.array(y_test)
    label_to_color = {0: cubehelix_colors[0], 1: cubehelix_colors[1]}
    point_colors = [label_to_color[label] for label in y_test_np]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=point_colors, alpha=0.5, edgecolor='none'
    )

    # Plot centroids with bigger markers and matching colors
    for idx, centroid in enumerate(centroids_2d):
        plt.scatter(
            centroid[0], centroid[1],
            s=200, color=cubehelix_colors[idx], marker='X', label=f'Centroid Class {classes[idx]}'
        )
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    legend_elements = [
        mpatches.Patch(facecolor=cubehelix_colors[0], label='Class 0'),
        mpatches.Patch(facecolor=cubehelix_colors[1], label='Class 1'),
        plt.Line2D([0], [0], marker='X', color='w', label='Centroid Class 0', markerfacecolor=cubehelix_colors[0], markersize=12),
        plt.Line2D([0], [0], marker='X', color='w', label='Centroid Class 1', markerfacecolor=cubehelix_colors[1], markersize=12),
    ]

    plt.legend(handles=legend_elements, title='Class')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    



    #====================================================================================================================
    # Plot Attention Mechanism
    #====================================================================================================================

    # 9. Visualize Categorical Attention 
    def create_attention_model(original_model):
        """Create a model that outputs attention weights from the last transformer block"""
        class AttentionModel(Model):
            def __init__(self, original_model):
                super().__init__()
                self.original_model = original_model
                
                # Store all categorical embeddings and transformer blocks
                self.cat_embeddings = [
                    layer for layer in original_model.layers 
                    if isinstance(layer, layers.Embedding) and 'embedding' in layer.name.lower()
                ]
                self.transformer_blocks = [
                    layer for layer in original_model.layers 
                    if isinstance(layer, TransformerBlock)
                ]
                
            def call(self, inputs):
                x_cat, x_num, x_text = inputs
                
                # Process categorical features
                embedded_cats = [emb_layer(x_cat[:, i:i+1]) for i, emb_layer in enumerate(self.cat_embeddings)]
                x_cat = layers.Concatenate(axis=1)(embedded_cats)
                
                # Process through all but last transformer block
                for block in self.transformer_blocks[:-1]:
                    x_cat = block(x_cat)
                
                # Get attention weights from last block
                _, attn_weights = self.transformer_blocks[-1].att(x_cat, x_cat, return_attention_scores=True)
                return attn_weights
        
        return AttentionModel(original_model)

    def plot_attention_weights(attn_weights, sample_idx, head_idx, cat_features):
        """Plot attention weights for a specific head"""
        plt.figure(figsize=(10, 8))
        head_weights = attn_weights[0, head_idx]
        
        # Center around mean and scale by standard deviation
        scaled_weights = (head_weights - np.mean(head_weights)) / np.std(head_weights)
        
        cubehelix_cmap = sns.cubehelix_palette(
            start=0.5, rot=-0.75, gamma=1.0, 
            dark=0.2, light=0.9, as_cmap=True
        )

        sns.heatmap(
            scaled_weights,
            annot=True, fmt=".1f",
            cmap=cubehelix_cmap,
            center=0,
            xticklabels=cat_features,
            yticklabels=cat_features,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        plt.title(f"Scaled Attention (σ) - Head {head_idx+1}\nSample {sample_idx}")
        plt.xticks(rotation=45)
        plt.show()

    def visualize_attention(model, sample_idx=0):
        """Visualize attention weights for a given sample with decoded categorical values"""
        # Categorical feature names and their possible values
        cat_features = ['Education', 'EmploymentType', 'MaritalStatus', 
                    'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
        
        # Define mappings from encoded values to actual labels
        category_mappings = {
            'Education': {
                0: 'High School', 
                1: 'Bachelor', 
                2: 'Master', 
                3: 'PhD'
            },
            'EmploymentType': {
                0: 'Unemployed', 
                1: 'Part-time', 
                2: 'Full-time', 
                3: 'Self-employed'
            },
            'MaritalStatus': {
                0: 'Single', 
                1: 'Married', 
                2: 'Divorced'
            },
            'HasMortgage': {
                0: 'No', 
                1: 'Yes'
            },
            'HasDependents': {
                0: 'No', 
                1: 'Yes'
            },
            'LoanPurpose': {
                0: 'Personal', 
                1: 'Education', 
                2: 'Medical', 
                3: 'Business'
            },
            'HasCoSigner': {
                0: 'No', 
                1: 'Yes'
            }
        }
        
        # Convert tensor to numpy array before accessing values
        categorical_values = X_test['categorical'][sample_idx].numpy()
        
        # Print the categorical values for this sample with decoded labels
        print("\nCategorical feature values for sample", sample_idx)
        for feat, val in zip(cat_features, categorical_values):
            decoded_value = category_mappings[feat].get(int(val), f"Unknown ({val})")
            print(f"{feat}: {decoded_value}")
        
        # Rest of the visualization code remains the same...
        attention_model = create_attention_model(model)
        
        sample = (
            tf.cast(tf.expand_dims(X_test['categorical'][sample_idx], 0), tf.int32),
            tf.cast(tf.expand_dims(X_test['numerical'][sample_idx], 0), tf.float32),
            tf.cast(tf.expand_dims(X_test['text'][sample_idx], 0), tf.int32)
        )
        
        attn_weights = attention_model.predict(sample, verbose=0)
        
        num_heads = attn_weights.shape[1]
        for head_idx in range(num_heads):
            plot_attention_weights(attn_weights, sample_idx, head_idx, cat_features)

    # Visualize for first 3 samples
    for i in range(min(3, len(X_test['categorical']))):
        print(f"\nVisualizing attention for sample {i}")
        visualize_attention(trained_model, i)





    # 10. Visualization of Text Attention
    def get_original_text(sample_idx):
        df = TabularDataPreprocessor.load_and_prepare_data("data/df_small_sampled.csv")
        _, test_data = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED)
        return test_data['JobDescription'].iloc[sample_idx]

    def get_text_vectorizer(model, train_text=None):
        try:
            vectorizer = model.get_layer('text_vectorizer')
            print("Using model's built-in text vectorizer")
            return vectorizer
        except ValueError:
            print("Creating new vectorizer with consistent preprocessing")

            def custom_standardize(text):
                text = tf.strings.lower(text)
                text = tf.strings.regex_replace(text, r"(\d+)%", r"\1 %")         
                text = tf.strings.regex_replace(text, r"[$€£]", "")              
                text = tf.strings.regex_replace(text, r"[^\w\s\-%]", " ")        
                text = tf.strings.regex_replace(text, r"\s+", " ")               
                return text

            vectorizer = layers.TextVectorization(
                max_tokens=MAX_TOKENS,
                output_mode='int',
                output_sequence_length=OUTPUT_SEQUENCE_LENGTH,
                standardize=custom_standardize,
                name='text_vectorizer'
            )

            if train_text is not None:
                vectorizer.adapt(train_text.fillna('').astype(str))
            return vectorizer

    def verify_tokenization(original_text, vectorizer):
        token_ids = vectorizer([original_text]).numpy()[0]
        vocab = vectorizer.get_vocabulary()
        tokens = [vocab[token_id] for token_id in token_ids if token_id != 0]
        labels = [f"{token} ({i+1})" for i, token in enumerate(tokens)]

        print("\nTokenization Verification:")
        print(f"Original: {' '.join(original_text.split()[:5])}...")
        print(f"Processed: {' '.join(tokens[:5])}...")

        if len(tokens) == 0:
            print("Warning: No tokens generated!")
        if any("%" in t for t in original_text.split()) and not any("%" in t for t in tokens):
            print("Warning: Percentage signs not preserved properly")

        return tokens, labels

    def create_text_attention_model(model):
        text_transformer = [l for l in model.layers if isinstance(l, TransformerBlock)][-1]
        _, attn_weights = text_transformer.att(
            model.get_layer('embedding_7').output,
            model.get_layer('embedding_7').output,
            return_attention_scores=True
        )
        return Model(inputs=model.inputs, outputs=attn_weights)

    def plot_text_attention(weights, tokens, sample_idx, head_idx):
        plt.figure(figsize=(20, 18))
        attn_df = pd.DataFrame(
            (weights - np.mean(weights)) / np.std(weights),
            index=tokens,
            columns=tokens
        )

        cubehelix_cmap = sns.cubehelix_palette(
            start=0.5, rot=-0.75, gamma=1.0,
            dark=0.2, light=0.9, as_cmap=True
        )

        ax = sns.heatmap(
            attn_df,
            cmap=cubehelix_cmap,
            center=0,
            annot=False,
            cbar_kws={'label': 'Normalized Attention (σ)'},
            linewidths=0.5
        )

        # Find important tokens with higher threshold (more selective)
        important_indices = set()
        threshold = 2.  # Increased from 1.5 to be more selective
        
        # Only mark tokens that have at least 2 strong connections
        for i in range(len(tokens)):
            strong_connections = 0
            for j in range(len(tokens)):
                if abs(attn_df.iloc[i,j]) > threshold and i != j:
                    strong_connections += 1
            if strong_connections >= 2:  # Require at least 2 strong connections
                important_indices.add(i)

        # Update label formatting (keeping font black)
        for label in ax.get_yticklabels():
            pos = int(label.get_text().split('(')[-1].rstrip(')')) - 1
            if pos in important_indices:
                label.set_weight('bold')
                label.set_color('black')  # Changed from darkred to black
                label.set_fontsize(10)
            else:
                label.set_fontsize(8)

        for label in ax.get_xticklabels():
            pos = int(label.get_text().split('(')[-1].rstrip(')')) - 1
            if pos in important_indices:
                label.set_weight('bold')
                label.set_color('black')  # Changed from darkred to black
                label.set_fontsize(10)
            else:
                label.set_fontsize(8)

        plt.title(
            f"Text Attention - Head {head_idx+1}\n"
            f"Sample {sample_idx} | Words: {len(tokens)}\n"
            "Bold labels show strongly attended tokens (σ > 2.5 with ≥2 connections)",
            pad=20, fontsize=14
        )
        plt.xlabel("Key Tokens (position)", fontsize=12)
        plt.ylabel("Query Tokens (position)", fontsize=12)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def prepare_attention_input(model, sample_idx, text_vectorizer):
        text = get_original_text(sample_idx)
        tokens, labels = verify_tokenization(text, text_vectorizer)
        token_ids = text_vectorizer([text]).numpy()[0]
        return {
            'inputs': {
                'categorical_inputs': tf.expand_dims(X_test['categorical'][sample_idx], 0),
                'numerical_inputs': tf.expand_dims(X_test['numerical'][sample_idx], 0),
                'text_inputs': tf.expand_dims(token_ids, 0)
            },
            'tokens': tokens,
            'labels': labels
        }

    def verify_token_alignment(original_text, vectorized_tokens, text_vectorizer):
        vocab = text_vectorizer.get_vocabulary()
        decoded_tokens = [vocab[token_id] for token_id in vectorized_tokens if token_id != 0]

        # Use same custom standardizer from vectorizer
        def custom_standardize_for_comparison(text):
            text = text.lower()
            text = re.sub(r"(\d+)%", r"\1 %", text)        # space before %
            text = re.sub(r"[$€£]", "", text)              # remove currency
            text = re.sub(r"[^\w\s\-%]", " ", text)        # keep alphanum, %, hyphen
            text = re.sub(r"\s+", " ", text)               # normalize whitespace
            return text.strip()

        # Apply the same preprocessing
        preprocessed_original = custom_standardize_for_comparison(original_text)
        original_tokens = preprocessed_original.split()

        print("\n=== Token Alignment Verification ===")
        print(f"Original: {original_tokens[:10]}... (Total: {len(original_tokens)})")
        print(f"Vectorized: {decoded_tokens[:10]}... (Total: {len(decoded_tokens)})")

        mismatch_found = False
        for i, (orig, vec) in enumerate(zip(original_tokens, decoded_tokens)):
            if orig != vec:
                print(f"\n🔍 First mismatch at position {i}:")
                print(f"Original word: '{orig}'")
                print(f"Processed token: '{vec}'")
                mismatch_found = True
                break

        if not mismatch_found and len(original_tokens) == len(decoded_tokens):
            print("✅ Perfect alignment!")
        elif not mismatch_found:
            print("⚠️ Lengths differ but all tokens match up to the shorter length")
        elif len(original_tokens) != len(decoded_tokens):
            print(f"\n⚠️ Length mismatch! Original: {len(original_tokens)} vs Vectorized: {len(decoded_tokens)}")

    def visualize_text_attention(model, sample_idx=0):
        try:
            df = TabularDataPreprocessor.load_and_prepare_data("data/df_small_sampled.csv")
            train_data, _ = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED)
            original_text = get_original_text(sample_idx)

            print(f"\n=== Original Text (Sample {sample_idx}) ===")
            print("-" * 80)
            print(original_text)
            print("-" * 80)

            text_vectorizer = get_text_vectorizer(model, train_data['JobDescription'])
            processed_tokens = text_vectorizer([original_text]).numpy()[0]
            verify_token_alignment(original_text, processed_tokens, text_vectorizer)

            vocab = text_vectorizer.get_vocabulary()
            decoded_tokens = [vocab[token_id] for token_id in processed_tokens if token_id != 0]
            clean_labels = [f"{token} ({i+1})" for i, token in enumerate(decoded_tokens)]

            attention_model = create_text_attention_model(model)
            sample_input = {
                'categorical_inputs': tf.expand_dims(X_test['categorical'][sample_idx], 0),
                'numerical_inputs': tf.expand_dims(X_test['numerical'][sample_idx], 0),
                'text_inputs': tf.expand_dims(processed_tokens, 0)
            }
            attn_weights = attention_model.predict(sample_input, verbose=0)[0]

            for head_idx in range(attn_weights.shape[0]):
                plot_text_attention(
                    attn_weights[head_idx][:len(decoded_tokens), :len(decoded_tokens)],
                    clean_labels,
                    sample_idx,
                    head_idx
                )

        except Exception as e:
            print(f"\nError visualizing sample {sample_idx}:")
            traceback.print_exc()

    # Run for first few samples
    for i in range(min(3, len(X_test['text']))):
        print(f"\n=== Analyzing Sample {i} ===")
        visualize_text_attention(trained_model, i)










    # %%

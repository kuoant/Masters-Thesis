#====================================================================================================================
# Imports and Constants
#====================================================================================================================
#%%

# Core Libraries
import numpy as np
import pandas as pd

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

# Machine Learning Models
import xgboost as xgb
    

# Constants
RANDOM_SEED = 42
TEST_SIZE = 0.2
MAX_TOKENS = 1000
OUTPUT_SEQUENCE_LENGTH = 20
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
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Class 0', 'Class 1'], 
                    yticklabels=['Class 0', 'Class 1'])
        plt.title('Model Confusion Matrix')
        plt.xlabel("Predicted")
        plt.ylabel("True")
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
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0', 'Class 1'], 
                yticklabels=['Class 0', 'Class 1'])
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("True")
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
        
        plt.figure(figsize=(8,6))
        sns.scatterplot(
            x=reduced_emb[:,0], y=reduced_emb[:,1],
            hue=labels,
            palette={0: 'red', 1: 'blue'},
            alpha=0.7
        )
        plt.title(f'{method.upper()} visualization of Transformer Embeddings')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend(title='Default')
        plt.show()

    # Plot embeddings with true labels
    plot_embeddings(test_embeddings, y_test.numpy(), method='tsne')

    # 8. Visualize Centroid with PCA
    classes = np.unique(y_test)
    centroids = np.array([test_embeddings[y_test == c].mean(axis=0) for c in classes])

    # Reduce centroids to 2D
    pca = PCA(n_components=2)
    centroids_2d = pca.fit_transform(centroids)
    embeddings_2d = pca.transform(test_embeddings)

    # Plot PCA & Centroids
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=embeddings_2d[:,0], y=embeddings_2d[:,1], hue=y_test.numpy(), 
                    palette={0:'red', 1:'blue'}, alpha=0.5)
    plt.scatter(centroids_2d[:,0], centroids_2d[:,1], s=200, c=['red','blue'], marker='X', label='Centroids')
    plt.title("PCA Embeddings with Class Centroids")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title='Class')
    plt.show()

    # 9. Plot Attention
    vocab = [
        '', '[UNK]', 'a', 'and', 'at', 'for', 'handling', 'office', 'managing', 'customer', 'support',
        'data', 'sales', 'in', 'services', 'parttime', 'months', 'hotel', 'service', 'representative',
        'inquiries', 'center', 'call', 'billing', 'supplies', 'schedules', 'invoices', 'administrator',
        'worker', 'warehouse', 'logistics', 'inventory', 'teacher', 'school', 'responsible', 'public',
        'planning', 'high', 'grading', 'curriculum', 'performance', 'interpreting', 'dashboards',
        'creating', 'analyst', 'writer', 'small', 'producing', 'materials', 'marketing', 'freelance',
        'content', 'businesses', 'store', 'retail', 'providing', 'clothing', 'associate', 'startup',
        'software', 'maintained', 'fintech', 'engineer', 'developed', 'backend', 'apis', 'worked',
        'primarily', 'local', 'hospitality', 'front', 'employed', 'desk', 'with', 'guest', 'assisting',
        'server', 'restaurant', 'evenings', 'as', 'part-time', 'services', 'over', 'years', 'since',
        'working', 'capacity', '60%', 'responsible', 'selling', 'holiday', 'trips', 'like', 'hostels',
        'tech-company', 'experience', '5-star', 'hotels', 'worldwide', '3', '2', '10', '12', '20', '25',
        '5', '7']

    vocab_size = len(vocab)

    # Sentence Input
    sentence = "worked part-time at a hotel for 12 months"
    word_to_id = {w: i for i, w in enumerate(vocab)}
    token_ids = [word_to_id.get(w, 1) for w in sentence.lower().split()]  # 1 = [UNK]

    # Pad to OUTPUT_SEQUENCE_LENGTH
    token_ids = token_ids + [0]*(OUTPUT_SEQUENCE_LENGTH - len(token_ids))
    sample_text = tf.expand_dims(token_ids[:OUTPUT_SEQUENCE_LENGTH], axis=0)

    # Dummy categorical and numerical input
    num_cat = len(cat_features_info)

    # Example mappings (replace with actual mappings from your preprocessing pipeline)
    education_map = {'High School': 0, 'Bachelors': 1, 'Masters': 2, 'PhD': 3}
    employment_map = {'Full-time': 0, 'Part-time': 1, 'Self-employed': 2}
    marital_map = {'Single': 0, 'Married': 1, 'Divorced': 2}
    mortgage_map = {'No': 0, 'Yes': 1}
    dependents_map = {'No': 0, 'Yes': 1}
    loanpurpose_map = {'Car': 0, 'Home': 1, 'Education': 2, 'Other': 3}
    cosigner_map = {'No': 0, 'Yes': 1}
        
    example_cat_values = [
    education_map['PhD'],
    employment_map['Part-time'],
    marital_map['Married'],
    mortgage_map['Yes'],
    dependents_map['Yes'],
    loanpurpose_map['Education'],
    cosigner_map['Yes']
    ]

    sample_cat = tf.constant([example_cat_values], dtype=tf.int32)

    sample_num = tf.zeros((1, num_numerical), dtype=tf.float32)

    # Text Attention Model
    text_inputs = layers.Input(shape=(OUTPUT_SEQUENCE_LENGTH,), name='text_inputs')
    text_embedding = layers.Embedding(vocab_size, EMBED_DIM, input_length=OUTPUT_SEQUENCE_LENGTH)(text_inputs)

    x = text_embedding
    attention_models_text = []
    for _ in range(NUM_TRANSFORMER_BLOCKS):
        block = TransformerBlock(EMBED_DIM, NUM_HEADS)
        attn_out, attn_weights = block.att(x, x, return_attention_scores=True)
        attention_models_text.append(
            Model(inputs=text_inputs, outputs=attn_weights)
        )
        x = block(x)

    # Then for prediction:
    all_text_attention = []
    for block_model in attention_models_text:
        all_text_attention.append(block_model.predict(sample_text))

    # Visualize Text attention
    token_words = [vocab[token] if token < len(vocab) else '[UNK]' for token in sample_text.numpy()[0]]
    non_empty_indices = [i for i, word in enumerate(token_words) if word != '']
    filtered_token_words = [token_words[i] for i in non_empty_indices]

    for block_idx, weights in enumerate(all_text_attention):
        num_heads = weights.shape[1]
        plt.figure(figsize=(20, 5))
        plt.suptitle(f"TEXT - Transformer Block {block_idx + 1}", y=1.02)

        for head_idx in range(num_heads):
            plt.subplot(1, num_heads, head_idx + 1)
            filtered_weights = weights[0, head_idx][non_empty_indices][:, non_empty_indices]
            sns.heatmap(filtered_weights, cmap="viridis",
                        xticklabels=filtered_token_words, yticklabels=filtered_token_words)
            plt.title(f"Head {head_idx + 1}")
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)

        plt.tight_layout()
        plt.show()


    # Categorical Attention Model
    categorical_inputs = layers.Input(shape=(num_cat,), name='categorical_inputs')

    # Create embeddings
    embedded_cats = []
    for i, (card, dim) in enumerate(cat_features_info):
        emb = layers.Embedding(input_dim=card, output_dim=dim)(categorical_inputs[:, i:i+1])
        embedded_cats.append(emb)
    x_cat = layers.Concatenate(axis=1)(embedded_cats)

    # Build the model that outputs attention weights at each block
    all_attention_outputs = []
    x = x_cat
    for i in range(NUM_TRANSFORMER_BLOCKS):
        transformer_block = TransformerBlock(EMBED_DIM, NUM_HEADS)
        # Get attention weights from this block
        attn_output, attn_weights = transformer_block.att(x, x, return_attention_scores=True)
        all_attention_outputs.append(attn_weights)
        # Pass through the full transformer block
        x = transformer_block(x)

    # Create model that outputs all attention weights
    attention_model = Model(
        inputs=categorical_inputs,
        outputs=all_attention_outputs
    )

    # Prepare sample input (using first sample from test set)
    sample_cat = X_test['categorical'][0:1]  # Take first sample and keep batch dimension

    # Get attention weights
    all_cat_attention = attention_model.predict(sample_cat)

    # Visualize Categorical attention
    cat_features = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 
                'HasDependents', 'LoanPurpose', 'HasCoSigner']

    for block_idx, weights in enumerate(all_cat_attention):
        num_heads = weights.shape[1]
        plt.figure(figsize=(15, 5))
        plt.suptitle(f"CATEGORICAL - Transformer Block {block_idx + 1}", y=1.02)

        for head_idx in range(num_heads):
            plt.subplot(1, num_heads, head_idx + 1)
            sns.heatmap(weights[0, head_idx], cmap="viridis",
                        xticklabels=cat_features, yticklabels=cat_features)
            plt.title(f"Head {head_idx + 1}")
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)

        plt.tight_layout()
        plt.show()


    # 10. Plot Default Rate Against Education

    # Ensure correct types
    df['Education'] = df['Education'].astype(str)
    df['HasDependents'] = df['HasDependents'].astype(str)
    df['Default'] = df['Default'].astype(int)

    # Group by Education, compute default rate
    grouped = df.groupby(['Education'])['Default'].agg(['sum', 'count'])
    grouped['default_rate'] = grouped['sum'] / grouped['count'] * 100  # in percent

    # Convert to 2D DataFrame for heatmap (1 column)
    heatmap_data = grouped[['default_rate']].sort_values('default_rate', ascending=False)

    # Sort the data by default rate (ascending for bottom-up bars)
    heatmap_data = heatmap_data.sort_values('default_rate', ascending=True)

    # Normalize default rate values to [0, 1] for colormap
    norm = plt.Normalize(heatmap_data['default_rate'].min(), heatmap_data['default_rate'].max())
    colors = cm.viridis(norm(heatmap_data['default_rate']))

    # Plot
    plt.figure(figsize=(8, 5))
    bars = plt.barh(
        heatmap_data.index,
        heatmap_data['default_rate'],
        color=colors
    )

    # Annotate values
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height() / 2, f'{width:.1f}%', va='center')

    # Labels and title
    plt.xlabel('Default Rate (%)')
    plt.title('Default Rate by Education (Dependents = Yes)')
    plt.tight_layout()
    plt.show()
# %%

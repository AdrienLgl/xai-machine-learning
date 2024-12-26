from sklearn.preprocessing import MinMaxScaler

class WichExplanationMakeSense:

    def __init__(self, model, X_train, y_train, background, feature_names, target_names):
        print("Starting...")
        self._dict_attr = {}
        self.model = model
        self.background = background # numpy array
        self.scaler = MinMaxScaler()
        self.feature_names = feature_names
        self.target_names = target_names
        self.X_train = X_train
        self.y_train = y_train
        self._init_explainers()

    def _init_explainers(self):
        print("Initializing explainers...")
        self._init_deeplift()
        self._init_anchor()
        self._init_deepshap()
        self._init_integrated_gradients()
        self._init_kernelshap()
        self._init_lime()
        self._init_sshap()
        self._init_surrogates()
    
    def model_predict(self, features):
        features = torch.tensor(features, dtype=torch.float32)
        y_pred = model(features).detach().numpy()  # Get model outputs (probabilities of class 1)
        prob_class_1 = y_pred[:, 0]  # Extract probability for class 1
        prob_class_0 = 1 - prob_class_1  # Compute probability for class 0
        return np.column_stack((prob_class_0, prob_class_1))  # Return probabilities for both classes

    def model_predict_with_rounded_output(self, features):
        features = torch.tensor(features, dtype=torch.float32)
        y_pred = model(features).detach().numpy()  # Get model outputs (probabilities of class 1)
        prob_class_1 = y_pred[:, 0]  # Extract probability for class 1
        prob_class_0 = 1 - prob_class_1  # Compute probability for class 0
        binary_class = np.where(prob_class_1 >= 0.5, 1, 0)
        return binary_class

    def normalize_scores(self, scores, feature_range=(0, 1)):
        scores = np.array(scores)
        scores[scores < 0] = 0
        min_val, max_val = np.min(scores), np.max(scores)
        range_min, range_max = feature_range    
        if max_val == min_val:
            return np.full_like(scores, range_min)    
        normalized = (scores - min_val) / (max_val - min_val)
        normalized = normalized * (range_max - range_min) + range_min
        return normalized

    def _init_deepshap(self):
        background = torch.tensor(self.background, dtype=torch.float32)
        self.deepshap_exp = shap.DeepExplainer(model, background)

    def compute_deepshap(self, features):
        assert features.shape[1] >= 1, "Features need to have 2D"
        print("Computing DeepSHAP...")
        features = torch.tensor(features, dtype=torch.float32).requires_grad_(True)
        attributions = self.deepshap_exp.shap_values(features)
        return self.normalize_scores(attributions)

    def _init_kernelshap(self):
        self.kernelshap_exp = shap.KernelExplainer(model=self.model_predict, data=self.background, feature_names=self.feature_names)

    def compute_kernelshap(self, features):
        assert features.shape[1] >= 1, "Features need to have 2D"
        print("Computing KernelSHAP...")
        attributions = np.array(self.kernelshap_exp.shap_values(features))[1]
        return self.normalize_scores(attributions)

    def _init_lime(self):
        self.lime_exp = LimeTabularExplainer(
            training_data=self.X_train,  # The training dataset (should be in same format as X_patient)
            feature_names=self.feature_names,
            class_names=self.target_names,  # Replace with your actual class names
            mode='classification',  # Since it's a classification problem
            training_labels=self.y_train,  # Your training labels (0 or 1)
        )

    def compute_lime(self, features):
        assert features.shape[1] >= 1, "Features need to have 2D"
        print("Computing LIME...")
        attributions = []
        for feat in features:
            explanation = self.lime_exp.explain_instance(
                feat.squeeze(),
                self.model_predict,  # The model's prediction function
                num_features=len(self.feature_names)  # Number of features to include in the explanation
            )
            attr = list(explanation.as_map().values())
            max_index = max(item[0] for row in attr for item in row)
            scores = np.zeros(max_index + 1)
            for row in attr:
                for index, value in row:
                    scores[index] = value
            attributions.append(scores)
        return self.normalize_scores(attributions)

    def _init_deeplift(self):
        self.deeplift_exp = DeepLift(self.model)

    def compute_deeplift(self, features):
        assert features.shape[1] >= 1, "Features need to have 2D"
        print("Computing DeepLift...")
        features = torch.tensor(features, dtype=torch.float32).requires_grad_(True)
        attributions, delta = self.deeplift_exp.attribute(features, target=0, return_convergence_delta=True)
        return self.normalize_scores(attributions.detach().numpy())

    def _init_anchor(self):
        self.anchor_exp = AnchorTabularExplainer(class_names=self.target_names, feature_names=self.feature_names, train_data=self.X_train)

    def compute_anchor(self, features):
        assert features.shape[1] >= 1, "Features need to have 2D"
        attributions = []
        for feat in features:
            anchor_explanation = self.anchor_exp.explain_instance(feat, self.model_predict_with_rounded_output, threshold=0.999)
            print(f"Anchor precision: {anchor_explanation.precision()}")
            print(f"Anchor coverage: {anchor_explanation.coverage()}")
            names = [cond.split()[0] for cond in anchor_explanation.names()]
            print(" AND ".join(anchor_explanation.names()))
            scores = [1 if col in names else 0 for col in self.feature_names]
            attributions.append(scores)
        return self.normalize_scores(attributions)
        
    def _init_sshap(self):
       self.sshap_exp = shap.explainers.Sampling(self.model_predict, self.X_train)

    def compute_sshap(self, features):
        assert features.shape[1] >= 1, "Features need to have 2D"
        print("Computing SSHAP...")
        attributions = self.sshap_exp.shap_values(features)[1] # classe 1
        return self.normalize_scores(attributions)

    def _init_integrated_gradients(self):
        self.ig_exp = IntegratedGradients(self.model)

    def compute_integrated_gradients(self, features):
        assert features.shape[1] >= 1, "Features need to have 2D"
        print("Computing Integrated Gradients...")
        features = torch.tensor(features, dtype=torch.float32).requires_grad_(True)
        attributions, delta = self.ig_exp.attribute(features, target=0, return_convergence_delta=True)
        return self.normalize_scores(attributions.detach().numpy())

    def _init_surrogates(self):
        self.surr_exp = MimicExplainer(
            self.model,
            self.X_train,
            SGDExplainableModel,
            augment_data=False,
            features=self.feature_names,
            model_task="classification"
        )

    def compute_global_surr(self, features):
        assert features.shape[1] >= 1, "Features need to have 2D"
        print("Computing Global Surrogates...")
        local_exp = self.surr_exp.explain_local(features)
        attributions = local_exp.local_importance_values[1] # classe 1
        return self.normalize_scores(attributions)
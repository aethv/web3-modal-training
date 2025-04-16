import axios from 'axios';

const API_URL = process.env.VUE_APP_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 seconds
});

export default {
  // Generate web3 code from a prompt
  generateCode(prompt, options = {}) {
    const requestData = {
      prompt,
      max_length: options.maxLength || 2048,
      temperature: options.temperature || 0.7,
      top_p: options.topP || 0.9,
      top_k: options.topK || 50
    };
    
    return apiClient.post('/generate', requestData);
  },
  
  // Submit feedback for improving the model
  submitFeedback(prompt, generatedCode, correctedCode, feedback = '', rating = null) {
    return apiClient.post('/feedback', {
      prompt,
      generated_code: generatedCode,
      corrected_code: correctedCode,
      feedback,
      rating
    });
  },
  
  // Get the current training status
  getTrainingStatus() {
    return apiClient.get('/training-status');
  },
  
  // Trigger model retraining
  triggerTraining() {
    return apiClient.post('/train');
  }
};
<template>
    <div class="training-container">
      <h3>Model Training Control</h3>
      <div class="training-status">
        <div class="status-indicator" :class="{ active: isTraining }"></div>
        <span>Status: {{ isTraining ? 'Training in progress...' : 'Idle' }}</span>
      </div>
      
      <div class="training-controls">
        <button 
          class="train-button" 
          @click="startTraining" 
          :disabled="isTraining"
        >
          Start Training
        </button>
        
        <div v-if="trainingProgress" class="progress-bar-container">
          <div class="progress-bar" :style="{ width: `${trainingProgress}%` }"></div>
          <span>{{ trainingProgress }}%</span>
        </div>
        
        <div v-if="trainingMessage" class="training-message">
          {{ trainingMessage }}
        </div>
      </div>
    </div>
  </template>
  
  <script>
  export default {
    name: 'TrainingControl',
    data() {
      return {
        isTraining: false,
        trainingProgress: 0,
        trainingMessage: '',
        trainingInterval: null
      };
    },
    beforeUnmount() {
      if (this.trainingInterval) {
        clearInterval(this.trainingInterval);
      }
    },
    methods: {
      async startTraining() {
        if (this.isTraining) return;
        
        this.isTraining = true;
        this.trainingProgress = 0;
        this.trainingMessage = 'Initializing training...';
        
        try {
          // Start the training process
          const response = await fetch('http://localhost:5000/train', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
              feedback_data: true,  // Include feedback data in training
              epochs: 5             // Optional: customize training parameters
            }),
          });
          
          const data = await response.json();
          
          if (data.status === 'started') {
            this.trainingMessage = 'Training started successfully';
            this.pollTrainingStatus();
          } else {
            this.trainingMessage = `Error: ${data.message || 'Unknown error'}`;
            this.isTraining = false;
          }
        } catch (error) {
          console.error('Error starting training:', error);
          this.trainingMessage = 'Error connecting to training server';
          this.isTraining = false;
        }
      },
      
      pollTrainingStatus() {
        // Poll the training status every 5 seconds
        this.trainingInterval = setInterval(async () => {
          try {
            const response = await fetch('http://localhost:5000/training_status');
            const data = await response.json();
            
            this.trainingProgress = data.progress || 0;
            
            if (data.status === 'completed') {
              this.trainingMessage = 'Training completed successfully!';
              this.isTraining = false;
              clearInterval(this.trainingInterval);
            } else if (data.status === 'failed') {
              this.trainingMessage = `Training failed: ${data.message}`;
              this.isTraining = false;
              clearInterval(this.trainingInterval);
            } else {
              this.trainingMessage = data.message || 'Training in progress...';
            }
          } catch (error) {
            console.error('Error checking training status:', error);
            this.trainingMessage = 'Error connecting to training server';
            clearInterval(this.trainingInterval);
            this.isTraining = false;
          }
        }, 5000);
      }
    }
  };
  </script>
  
  <style>
  .training-container {
    background-color: #f7f9fa;
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    margin-top: 20px;
  }
  
  .training-status {
    display: flex;
    align-items: center;
    margin-bottom: 15px;
  }
  
  .status-indicator {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background-color: #95a5a6;
    margin-right: 8px;
  }
  
  .status-indicator.active {
    background-color: #2ecc71;
    animation: pulse 1.5s infinite;
  }
  
  @keyframes pulse {
    0% { opacity: 0.6; }
    50% { opacity: 1; }
    100% { opacity: 0.6; }
  }
  
  .train-button {
    padding: 10px 20px;
    background-color: #3498db;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 16px;
  }
  
  .train-button:hover {
    background-color: #2980b9;
  }
  
  .train-button:disabled {
    background-color: #95a5a6;
    cursor: not-allowed;
  }
  
  .progress-bar-container {
    margin-top: 15px;
    height: 20px;
    background-color: #ecf0f1;
    border-radius: 10px;
    overflow: hidden;
    position: relative;
  }
  
  .progress-bar {
    height: 100%;
    background-color: #2ecc71;
    transition: width 0.5s ease;
  }
  
  .progress-bar-container span {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: #2c3e50;
    font-weight: bold;
    font-size: 12px;
  }
  
  .training-message {
    margin-top: 10px;
    padding: 8px;
    border-radius: 4px;
    background-color: #ecf0f1;
    color: #2c3e50;
    font-size: 14px;
  }
  </style>
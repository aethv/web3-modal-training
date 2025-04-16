<template>
  <div class="training-status">
    <h3>Model Training Status</h3>
    
    <div class="status-card">
      <div class="status-header">
        <div class="status-indicator" :class="statusClass"></div>
        <div class="status-title">{{ formattedStatus }}</div>
      </div>
      
      <div class="status-details" v-if="status">
        <div class="status-item">
          <span class="label">Last Update:</span>
          <span class="value">{{ status.last_update }}</span>
        </div>
        
        <div class="status-item" v-if="status.current_epoch && status.total_epochs">
          <span class="label">Progress:</span>
          <div class="progress-bar">
            <div 
              class="progress-value" 
              :style="{width: `${(status.current_epoch / status.total_epochs) * 100}%`}"
            ></div>
          </div>
          <span class="progress-text">{{ status.current_epoch }} / {{ status.total_epochs }} epochs</span>
        </div>
        
        <div class="status-item" v-if="status.loss">
          <span class="label">Current Loss:</span>
          <span class="value">{{ typeof status.loss === 'number' ? status.loss.toFixed(4) : status.loss }}</span>
        </div>
      </div>
      
      <div class="status-actions">
        <button 
          @click="triggerTraining" 
          class="btn btn-primary" 
          :disabled="isTrainingActive || triggering"
        >
          {{ triggering ? 'Triggering...' : 'Trigger Training' }}
        </button>
        
        <button @click="refreshStatus" class="btn btn-secondary">
          Refresh Status
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import api from '../services/api';

export default {
  name: 'TrainingStatus',
  data() {
    return {
      status: null,
      loading: true,
      error: null,
      triggering: false,
      refreshInterval: null
    };
  },
  computed: {
    isTrainingActive() {
      return this.status && ['preparing', 'training'].includes(this.status.status);
    },
    statusClass() {
      if (!this.status) return 'status-unknown';
      
      switch (this.status.status) {
        case 'idle': return 'status-idle';
        case 'preparing': return 'status-preparing';
        case 'training': return 'status-training';
        case 'completed': return 'status-completed';
        case 'failed': return 'status-failed';
        default: return 'status-unknown';
      }
    },
    formattedStatus() {
      if (!this.status) return 'Unknown';
      
      switch (this.status.status) {
        case 'idle': return 'Idle';
        case 'preparing': return 'Preparing Data';
        case 'training': return 'Training In Progress';
        case 'completed': return 'Training Completed';
        case 'failed': return 'Training Failed';
        default: return 'Unknown';
      }
    }
  },
  created() {
    this.fetchStatus();
    
    // Set up interval to refresh status every 30 seconds
    this.refreshInterval = setInterval(this.fetchStatus, 30000);
  },
  beforeUnmount() {
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval);
    }
  },
  methods: {
    async fetchStatus() {
      try {
        this.loading = true;
        const response = await api.getTrainingStatus();
        this.status = response.data;
        this.error = null;
      } catch (err) {
        console.error('Error fetching training status:', err);
        this.error = 'Failed to fetch training status';
      } finally {
        this.loading = false;
      }
    },
    async triggerTraining() {
      if (this.isTrainingActive) return;
      
      try {
        this.triggering = true;
        await api.triggerTraining();
        
        // Refresh status after triggering
        await this.fetchStatus();
      } catch (err) {
        console.error('Error triggering training:', err);
        alert('Failed to trigger training. Please try again.');
      } finally {
        this.triggering = false;
      }
    },
    refreshStatus() {
      this.fetchStatus();
    }
  }
};
</script>

<style scoped>
.training-status {
  margin-top: 30px;
  padding: 20px;
  background-color: #f8f9fa;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

h3 {
  margin-top: 0;
  margin-bottom: 15px;
  color: #333;
}

.status-card {
  background-color: white;
  border-radius: 6px;
  padding: 15px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.status-header {
  display: flex;
  align-items: center;
  margin-bottom: 15px;
}

.status-indicator {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  margin-right: 10px;
}

.status-idle {
  background-color: #9aa0a6;
}

.status-preparing {
  background-color: #fbbc04;
}

.status-training {
  background-color: #4285f4;
}

.status-completed {
  background-color: #0f9d58;
}

.status-failed {
  background-color: #d93025;
}

.status-unknown {
  background-color: #9aa0a6;
}

.status-title {
  font-weight: bold;
  font-size: 18px;
}

.status-details {
  margin-bottom: 20px;
}

.status-item {
  margin-bottom: 10px;
}

.label {
  font-weight: 500;
  margin-right: 10px;
  color: #5f6368;
}

.progress-bar {
  height: 8px;
  background-color: #e8eaed;
  border-radius: 4px;
  margin: 8px 0;
  overflow: hidden;
}

.progress-value {
  height: 100%;
  background-color: #4285f4;
  transition: width 0.3s ease;
}

.progress-text {
  font-size: 14px;
  color: #5f6368;
}
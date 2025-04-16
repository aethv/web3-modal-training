<template>
  <div class="feedback-form-overlay" v-if="show">
    <div class="feedback-form">
      <h2>Submit Feedback</h2>
      <p>Your feedback helps our model learn and improve.</p>
      
      <div class="form-group">
        <label>Original Prompt:</label>
        <div class="prompt-display">{{ feedbackData.prompt }}</div>
      </div>
      
      <div class="form-group">
        <label>Your Rating:</label>
        <div class="rating">
          <span 
            v-for="star in 5" 
            :key="star"
            class="star"
            :class="{ active: star <= rating }"
            @click="setRating(star)"
          >
            ★
          </span>
        </div>
      </div>
      
      <div class="form-group">
        <label for="feedback-text">Additional Comments:</label>
        <textarea 
          id="feedback-text"
          v-model="feedbackText"
          placeholder="What was good or bad about the generated code? Any suggestions for improvement?"
          rows="4"
          class="form-control"
        ></textarea>
      </div>
      
      <div class="form-actions">
        <button @click="close" class="btn btn-secondary">Cancel</button>
        <button @click="submitFeedback" class="btn btn-primary" :disabled="submitting">
          {{ submitting ? 'Submitting...' : 'Submit Feedback' }}
        </button>
      </div>
      
      <div class="submission-status" v-if="submissionStatus">
        <p :class="submissionStatus.type">{{ submissionStatus.message }}</p>
      </div>
    </div>
  </div>
</template>

<script>
import api from '../services/api';

export default {
  name: 'FeedbackForm',
  props: {
    show: {
      type: Boolean,
      default: false
    },
    feedbackData: {
      type: Object,
      default: () => ({
        prompt: '',
        generatedCode: '',
        correctedCode: ''
      })
    }
  },
  emits: ['close', 'feedback-submitted'],
  data() {
    return {
      rating: 0,
      feedbackText: '',
      submitting: false,
      submissionStatus: null
    };
  },
  methods: {
    setRating(star) {
      this.rating = star;
    },
    close() {
      this.$emit('close');
    },
    async submitFeedback() {
      this.submitting = true;
      this.submissionStatus = null;
      
      try {
        await api.submitFeedback(
          this.feedbackData.prompt,
          this.feedbackData.generatedCode,
          this.feedbackData.correctedCode,
          this.feedbackText,
          this.rating
        );
        
        this.submissionStatus = {
          type: 'success',
          message: 'Feedback submitted successfully. Thank you!'
        };
        
        // Reset form
        this.rating = 0;
        this.feedbackText = '';
        
        // Emit event
        this.$emit('feedback-submitted');
        
        // Close after a delay
        setTimeout(() => {
          this.close();
        }, 2000);
        
      } catch (err) {
        console.error('Feedback submission error:', err);
        this.submissionStatus = {
          type: 'error',
          message: 'Failed to submit feedback. Please try again.'
        };
      } finally {
        this.submitting = false;
      }
    }
  }
};
</script>

<style scoped>
.feedback-form-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.feedback-form {
  width: 90%;
  max-width: 600px;
  background-color: white;
  border-radius: 8px;
  padding: 25px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

h2 {
  margin-top: 0;
  margin-bottom: 10px;
}

p {
  margin-bottom: 20px;
  color: #666;
}

.form-group {
  margin-bottom: 20px;
}

label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
}

.prompt-display {
  padding: 10px;
  background-color: #f5f5f5;
  border-radius: 4px;
  min-height: 40px;
  max-height: 100px;
  overflow-y: auto;
  font-family: monospace;
  color: #333;
}

.rating {
  display: flex;
  gap: 5px;
}

.star {
  font-size: 24px;
  color: #ddd;
  cursor: pointer;
}

.star.active {
  color: #ffc107;
}

.form-control {
  width: 100%;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-family: inherit;
  font-size: 16px;
}

.form-actions {
  display: flex;
  gap: 10px;
  justify-content: flex-end;
}

.btn {
  padding: 10px 20px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-weight: bold;
}

.btn-primary {
  background-color: #4285f4;
  color: white;
}

.btn-primary:disabled {
  background-color: #a1c0f7;
  cursor: not-allowed;
}

.btn-secondary {
  background-color: #f1f3f4;
  color: #202124;
}

.submission-status {
  margin-top: 15px;
  text-align: center;
}

.success {
  color: #0f9d58;
}

.error {
  color: #d93025;
}
</style>
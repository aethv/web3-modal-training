<template>
  <div class="feedback-container">
    <button 
      class="toggle-feedback-btn"
      @click="isExpanded = !isExpanded"
    >
      {{ isExpanded ? 'Hide Feedback Form' : 'Provide Feedback' }}
    </button>
    
    <form v-if="isExpanded" @submit.prevent="submitFeedback" class="feedback-form">
      <div class="rating-container">
        <label>Rate the generated code (1-5):</label>
        <div class="star-rating">
          <span 
            v-for="star in 5" 
            :key="star"
            :class="['star', star <= rating ? 'filled' : '']"
            @click="rating = star"
          >
            ★
          </span>
        </div>
      </div>
      
      <div class="text-feedback">
        <label>Your feedback:</label>
        <textarea
          v-model="feedbackText"
          placeholder="What could be improved?"
          rows="3"
        ></textarea>
      </div>
      
      <div class="code-improvement">
        <label>Suggest code improvements:</label>
        <textarea
          v-model="updatedCode"
          placeholder="Edit the code to suggest improvements"
          rows="10"
        ></textarea>
      </div>
      
      <button type="submit" class="submit-feedback-btn">
        Submit Feedback
      </button>
    </form>
  </div>
</template>

<script>
export default {
  name: 'FeedbackForm',
  props: {
    originalCode: {
      type: String,
      required: true
    }
  },
  data() {
    return {
      updatedCode: this.originalCode,
      feedbackText: '',
      rating: 3,
      isExpanded: false
    };
  },
  watch: {
    originalCode(newValue) {
      this.updatedCode = newValue;
    }
  },
  methods: {
    submitFeedback() {
      this.$emit('submit', {
        updatedCode: this.updatedCode,
        feedbackText: this.feedbackText,
        rating: this.rating
      });
      
      // Reset form
      this.feedbackText = '';
      this.rating = 3;
      this.isExpanded = false;
    }
  }
};
</script>

<style>
.feedback-container {
  margin-bottom: 20px;
}

.toggle-feedback-btn {
  padding: 10px 20px;
  background-color: #2ecc71;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
  margin-bottom: 15px;
}

.toggle-feedback-btn:hover {
  background-color: #27ae60;
}

.feedback-form {
  background-color: #f7f9fa;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.rating-container, .text-feedback, .code-improvement {
  margin-bottom: 15px;
}

.star-rating {
  display: flex;
  gap: 5px;
  margin-top: 5px;
}

.star {
  font-size: 24px;
  color: #ccc;
  cursor: pointer;
}

.star.filled {
  color: #f1c40f;
}

textarea {
  width: 100%;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-family: inherit;
  font-size: 14px;
  resize: vertical;
}

.submit-feedback-btn {
  padding: 10px 20px;
  background-color: #3498db;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
}

.submit-feedback-btn:hover {
  background-color: #2980b9;
}

label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
}
</style>
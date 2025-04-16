<template>
  <div class="app-container">
    <header>
      <h1>Web3 Application Generator</h1>
      <p>Enter a prompt to generate a web3 application</p>
    </header>

    <div class="main-content">
      <div class="sidebar">
        <HistoryView 
          :history="promptHistory" 
          @select-item="loadHistoryItem" 
        />
        
        <!-- Add training control component -->
        <TrainingControl v-if="showTrainingControls" />
      </div>
      
      <div class="content-area">
        <PromptInput 
          @submit="handleGenerateCode" 
          :isLoading="isLoading" 
          :initialPrompt="selectedHistoryItem?.prompt || ''"
        />
        
        <div v-if="generatedCode">
          <CodePreview :code="generatedCode" />
          <FeedbackForm 
            :originalCode="generatedCode" 
            @submit="handleFeedbackSubmit" 
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import CodePreview from './components/CodePreview.vue';
import FeedbackForm from './components/FeedbackForm.vue';
import HistoryView from './components/HistoryView.vue';
import PromptInput from './components/PromptInput.vue';
import TrainingControl from './components/TrainingControl.vue';

export default {
  name: 'App',
  components: {
    PromptInput,
    CodePreview,
    FeedbackForm,
    HistoryView,
    TrainingControl
  },
  data() {
    return {
      generatedCode: '',
      isLoading: false,
      promptHistory: [],
      selectedHistoryItem: null,
      showTrainingControls: false // Set to true for admin users or toggle with admin login
    };
  },
  created() {
    // Load history from localStorage on initial render
    const savedHistory = localStorage.getItem('promptHistory');
    if (savedHistory) {
      this.promptHistory = JSON.parse(savedHistory);
    }
    
    // Check if admin mode should be enabled
    const isAdmin = localStorage.getItem('isAdmin') === 'true';
    this.showTrainingControls = isAdmin;
    
    // For demonstration, allow enabling admin mode with URL parameter
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('admin') === 'true') {
      this.showTrainingControls = true;
      localStorage.setItem('isAdmin', 'true');
    }
  },
  watch: {
    // Save history to localStorage whenever it changes
    promptHistory: {
      handler(newHistory) {
        localStorage.setItem('promptHistory', JSON.stringify(newHistory));
      },
      deep: true
    }
  },
  methods: {
    async handleGenerateCode(prompt) {
      this.isLoading = true;
      try {
        const response = await fetch('http://localhost:5000/generate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ prompt }),
        });
        
        const data = await response.json();
        this.generatedCode = data.generated_code;
        
        // Add to history
        const newHistoryItem = {
          id: Date.now(),
          prompt,
          code: data.generated_code,
          timestamp: new Date().toISOString(),
        };
        
        this.promptHistory = [newHistoryItem, ...this.promptHistory];
      } catch (error) {
        console.error('Error generating code:', error);
        alert('Error generating code. Please try again.');
      } finally {
        this.isLoading = false;
      }
    },
    async handleFeedbackSubmit(feedback) {
      try {
        await fetch('http://localhost:5000/feedback', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            originalCode: this.generatedCode,
            updatedCode: feedback.updatedCode,
            feedbackText: feedback.feedbackText,
            rating: feedback.rating
          }),
        });
        
        alert('Feedback submitted successfully!');
      } catch (error) {
        console.error('Error submitting feedback:', error);
        alert('Error submitting feedback. Please try again.');
      }
    },
    loadHistoryItem(item) {
      this.selectedHistoryItem = item;
      this.generatedCode = item.code;
    }
  }
};
</script>

<style>
.app-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  font-family: 'Arial', sans-serif;
}

header {
  text-align: center;
  margin-bottom: 30px;
}

header h1 {
  color: #2c3e50;
  margin-bottom: 10px;
}

.main-content {
  display: flex;
  gap: 20px;
}

.sidebar {
  flex: 0 0 250px;
}

.content-area {
  flex: 1;
}

@media (max-width: 768px) {
  .main-content {
    flex-direction: column;
  }
  
  .sidebar {
    flex: none;
    width: 100%;
  }
}
</style>
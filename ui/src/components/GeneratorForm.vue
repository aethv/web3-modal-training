<template>
  <div class="generator-form">
    <h2>Generate Web3 App</h2>
    
    <div class="form-group">
      <label for="prompt">Describe your Web3 application:</label>
      <textarea 
        id="prompt"
        v-model="prompt"
        placeholder="Example: Create a simple ERC-20 token with name 'MyToken', symbol 'MTK', and total supply of 1 million"
        rows="5"
        class="form-control"
      ></textarea>
    </div>
    
    <div class="generation-options" v-show="showAdvancedOptions">
      <h3>Advanced Options</h3>
      <div class="form-group">
        <label for="temperature">Temperature:</label>
        <input 
          type="range" 
          id="temperature" 
          v-model.number="temperature" 
          min="0.1" 
          max="1.0" 
          step="0.1"
        >
        <span>{{ temperature }}</span>
      </div>
      
      <div class="form-group">
        <label for="maxLength">Maximum Length:</label>
        <input 
          type="number" 
          id="maxLength" 
          v-model.number="maxLength" 
          min="512" 
          max="4096"
        >
      </div>
    </div>
    
    <div class="form-actions">
      <button 
        @click="toggleAdvancedOptions" 
        class="btn btn-secondary"
      >
        {{ showAdvancedOptions ? 'Hide' : 'Show' }} Advanced Options
      </button>
      
      <button 
        @click="generateCode" 
        class="btn btn-primary" 
        :disabled="isGenerating || !prompt.trim()"
      >
        {{ isGenerating ? 'Generating...' : 'Generate Code' }}
      </button>
    </div>
    
    <div class="generation-status" v-if="isGenerating">
      <div class="spinner"></div>
      <p>Generating your Web3 application... This may take a moment.</p>
    </div>
    
    <div class="generation-error" v-if="error">
      <p class="error">{{ error }}</p>
    </div>
  </div>
</template>

<script>
import api from '../services/api';

export default {
  name: 'GeneratorForm',
  data() {
    return {
      prompt: '',
      temperature: 0.7,
      maxLength: 2048,
      topP: 0.9,
      topK: 50,
      showAdvancedOptions: false,
      isGenerating: false,
      error: null
    };
  },
  methods: {
    toggleAdvancedOptions() {
      this.showAdvancedOptions = !this.showAdvancedOptions;
    },
    async generateCode() {
      if (!this.prompt.trim()) return;
      
      this.isGenerating = true;
      this.error = null;
      
      try {
        const options = {
          temperature: this.temperature,
          maxLength: this.maxLength,
          topP: this.topP,
          topK: this.topK
        };
        
        const response = await api.generateCode(this.prompt, options);
        
        // Emit the generated code to parent component
        this.$emit('code-generated', {
          prompt: this.prompt,
          code: response.data.code,
          generationTime: response.data.generation_time
        });
      } catch (err) {
        console.error('Generation error:', err);
        this.error = err.response?.data?.detail || 'An error occurred during code generation.';
      } finally {
        this.isGenerating = false;
      }
    }
  }
};
</script>

<style scoped>
.generator-form {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
  background-color: #f8f9fa;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

h2 {
  margin-bottom: 20px;
  color: #333;
}

.form-group {
  margin-bottom: 20px;
}

label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
}

.form-control {
  width: 100%;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-family: inherit;
  font-size: 16px;
}

.generation-options {
  margin: 20px 0;
  padding: 15px;
  background-color: #eef1f5;
  border-radius: 4px;
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

.generation-status {
  display: flex;
  align-items: center;
  margin-top: 20px;
}

.spinner {
  width: 24px;
  height: 24px;
  border: 3px solid #ddd;
  border-top: 3px solid #4285f4;
  border-radius: 50%;
  margin-right: 10px;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.error {
  color: #d93025;
  margin-top: 15px;
}
</style>
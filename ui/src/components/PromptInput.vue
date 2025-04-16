<template>
    <div class="prompt-input-container">
      <form @submit.prevent="submitPrompt">
        <textarea
          class="prompt-textarea"
          v-model="promptText"
          placeholder="Describe the web3 application you want to generate..."
          rows="5"
          required
        ></textarea>
        <button 
          type="submit" 
          class="generate-button"
          :disabled="isLoading || !promptText.trim()"
        >
          {{ isLoading ? 'Generating...' : 'Generate Web3 App' }}
        </button>
      </form>
    </div>
  </template>
  
  <script>
  export default {
    name: 'PromptInput',
    props: {
      isLoading: {
        type: Boolean,
        default: false
      },
      initialPrompt: {
        type: String,
        default: ''
      }
    },
    data() {
      return {
        promptText: this.initialPrompt
      };
    },
    watch: {
      initialPrompt(newValue) {
        this.promptText = newValue;
      }
    },
    methods: {
      submitPrompt() {
        if (this.promptText.trim()) {
          this.$emit('submit', this.promptText);
        }
      }
    }
  };
  </script>
  
  <style>
  .prompt-input-container {
    margin-bottom: 20px;
    background-color: #f7f9fa;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }
  
  .prompt-textarea {
    width: 100%;
    padding: 12px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 16px;
    font-family: inherit;
    resize: vertical;
  }
  
  .generate-button {
    margin-top: 10px;
    padding: 12px 20px;
    background-color: #3498db;
    color: white;
    border: none;
    border-radius: 4px;
    font-size: 16px;
    cursor: pointer;
    transition: background-color 0.3s;
  }
  
  .generate-button:hover {
    background-color: #2980b9;
  }
  
  .generate-button:disabled {
    background-color: #95a5a6;
    cursor: not-allowed;
  }
  </style>
<template>
  <div class="code-preview">
    <h2>Generated Code</h2>
    
    <div class="stats" v-if="generationTime">
      <span>Generated in: {{ generationTime.toFixed(2) }}s</span>
    </div>
    
    <div class="editor-container">
      <MonacoEditor
        v-model:value="codeValue"
        :language="detectLanguage()"
        :options="editorOptions"
        @change="onChange"
      />
    </div>
    
    <div class="actions">
      <button @click="copyCode" class="btn btn-secondary">
        Copy Code
      </button>
      <button @click="submitFeedback" class="btn btn-primary">
        Submit Feedback
      </button>
    </div>
    
    <div v-if="copied" class="copy-notification">
      Code copied to clipboard!
    </div>
  </div>
</template>

<script>
import MonacoEditor from '@/components/MonacoEditor.vue';
export default {
  name: 'CodePreview',
  components: {
    MonacoEditor
  },
  props: {
    code: {
      type: String,
      default: ''
    },
    prompt: {
      type: String,
      default: ''
    },
    generationTime: {
      type: Number,
      default: null
    }
  },
  data() {
    return {
      codeValue: this.code,
      copied: false,
      editorOptions: {
        theme: 'vs-dark',
        automaticLayout: true,
        fontSize: 14,
        scrollBeyondLastLine: false,
        minimap: { enabled: true },
        scrollbar: {
          verticalScrollbarSize: 10,
          horizontalScrollbarSize: 10
        }
      }
    };
  },
  watch: {
    code(newCode) {
      this.codeValue = newCode;
    }
  },
  methods: {
    detectLanguage() {
      // Simple language detection based on file extension patterns in the code
      const code = this.codeValue.toLowerCase();
      
      if (code.includes('pragma solidity') || code.includes('contract ')) {
        return 'solidity';
      } else if (code.includes('import react') || code.includes('export default') || code.includes('function(') || code.includes('=>')) {
        return 'javascript';
      } else if (code.includes('import "web3"') || code.includes('eth_')) {
        return 'javascript';
      } else if (code.includes('#include') || code.includes('int main')) {
        return 'cpp';
      } else if (code.includes('def ') || code.includes('import ') && code.includes(':')) {
        return 'python';
      }
      
      return 'javascript'; // Default to JavaScript
    },
    onChange(value) {
      this.codeValue = value;
    },
    copyCode() {
      navigator.clipboard.writeText(this.codeValue)
        .then(() => {
          this.copied = true;
          setTimeout(() => {
            this.copied = false;
          }, 2000);
        })
        .catch(err => {
          console.error('Failed to copy code', err);
        });
    },
    submitFeedback() {
      // Emit event to show feedback form
      this.$emit('show-feedback', {
        prompt: this.prompt,
        generatedCode: this.code,
        correctedCode: this.codeValue
      });
    }
  }
};
</script>

<style>
.code-preview {
  margin-top: 30px;
  padding: 20px;
  background-color: #f8f9fa;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.code-preview-container {
  margin-bottom: 20px;
  background-color: #f7f9fa;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  overflow: hidden;
}

.code-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 20px;
  background-color: #2c3e50;
  color: white;
}

.code-header h2 {
  margin: 0;
  font-size: 18px;
}

.copy-button {
  padding: 6px 12px;
  background-color: #3498db;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.copy-button:hover {
  background-color: #2980b9;
}

.code-block {
  padding: 20px;
  margin: 0;
  background-color: #282c34;
  color: #abb2bf;
  overflow-x: auto;
  font-family: 'Courier New', Courier, monospace;
  font-size: 14px;
  line-height: 1.5;
}
</style>
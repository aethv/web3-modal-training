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

<style scoped>
.code-preview {
  margin-top: 30px;
  padding: 20px;
  background-color: #f8f9fa;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

h2 {
  margin-bottom: 15px;
  color: #333;
}

.stats {
  margin-bottom: 10px;
  color: #666;
  font-size: 14px;
}

.editor-container {
  height: 500px;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 15px;
}

.actions {
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

.btn-secondary {
  background-color: #f1f3f4;
  color: #202124;
}

.copy-notification {
  position: fixed;
  bottom: 20px;
  right: 20px;
  background-color: #323232;
  color: white;
  padding: 10px 20px;
  border-radius: 4px;
  z-index: 1000;
}
</style>
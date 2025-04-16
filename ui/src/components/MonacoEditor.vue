<template>
  <div ref="monaco" class="monaco-editor-container"></div>
</template>

<script>
import * as monaco from 'monaco-editor';
import { onMounted, onBeforeUnmount, watch } from 'vue';

export default {
  name: 'MonacoEditor',
  props: {
    value: {
      type: String,
      default: ''
    },
    language: {
      type: String,
      default: 'javascript'
    },
    options: {
      type: Object,
      default: () => ({})
    }
  },
  emits: ['change', 'update:value'],
  setup(props, { emit, refs }) {
    let editor = null;
    let subscription = null;

    onMounted(() => {
      editor = monaco.editor.create(refs.monaco, {
        value: props.value,
        language: props.language,
        ...props.options
      });

      // Handle editor changes
      subscription = editor.onDidChangeModelContent(() => {
        const value = editor.getValue();
        emit('change', value);
        emit('update:value', value);
      });

      // Register Solidity language if not already registered
      if (!monaco.languages.getLanguages().some(lang => lang.id === 'solidity')) {
        monaco.languages.register({ id: 'solidity' });
        monaco.languages.setMonarchTokensProvider('solidity', {
          tokenizer: {
            root: [
              [/pragma\s+solidity/, 'keyword'],
              [/contract|library|interface/, 'keyword'],
              [/function|returns|external|public|private|internal|pure|view/, 'keyword'],
              [/mapping|address|uint|int|bool|string|bytes/, 'type'],
              [/(\/\/.*$)/, 'comment'],
              [/(\/\*[\s\S]*?\*\/)/, 'comment'],
              [/".*?"/, 'string'],
              [/'.*?'/, 'string'],
              [/\d+/, 'number'],
              [/[{}()\[\]]/, '@brackets'],
              [/[;,.]/, 'delimiter'],
            ]
          }
        });
      }
    });

    onBeforeUnmount(() => {
      if (subscription) {
        subscription.dispose();
      }
      if (editor) {
        editor.dispose();
      }
    });

    watch(() => props.value, (newValue) => {
      if (editor && newValue !== editor.getValue()) {
        editor.setValue(newValue);
      }
    });

    watch(() => props.language, (newLang) => {
      if (editor) {
        monaco.editor.setModelLanguage(editor.getModel(), newLang);
      }
    });

    watch(() => props.options, (newOptions) => {
      if (editor) {
        editor.updateOptions(newOptions);
      }
    }, { deep: true });

    return {};
  }
};
</script>

<style scoped>
.monaco-editor-container {
  height: 100%;
  width: 100%;
}
</style>
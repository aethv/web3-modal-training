<template>
    <div class="history-container" :class="{ empty: history.length === 0 }">
      <h3>Prompt History</h3>
      <p v-if="history.length === 0">No history yet. Generate your first web3 app!</p>
      <ul v-else class="history-list">
        <li 
          v-for="item in history" 
          :key="item.id" 
          class="history-item"
          @click="$emit('select-item', item)"
        >
          <div class="history-prompt">{{ truncateText(item.prompt, 50) }}</div>
          <div class="history-date">
            {{ new Date(item.timestamp).toLocaleString() }}
          </div>
        </li>
      </ul>
    </div>
  </template>
  
  <script>
  export default {
    name: 'HistoryView',
    props: {
      history: {
        type: Array,
        default: () => []
      }
    },
    methods: {
      truncateText(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
      }
    }
  };
  </script>
  
  <style>
  .history-container {
    background-color: #f7f9fa;
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }
  
  .history-container h3 {
    margin-top: 0;
    margin-bottom: 15px;
    color: #2c3e50;
  }
  
  .history-list {
    list-style-type: none;
    padding: 0;
    margin: 0;
    max-height: 500px;
    overflow-y: auto;
  }
  
  .history-item {
    padding: 10px;
    border-bottom: 1px solid #ddd;
    cursor: pointer;
    transition: background-color 0.2s;
  }
  
  .history-item:hover {
    background-color: #ecf0f1;
  }
  
  .history-item:last-child {
    border-bottom: none;
  }
  
  .history-prompt {
    font-weight: bold;
    margin-bottom: 5px;
  }
  
  .history-date {
    font-size: 12px;
    color: #7f8c8d;
  }
  
  .empty p {
    color: #7f8c8d;
    font-style: italic;
  }
  </style>
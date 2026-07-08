const express = require('express');
const app = express();
const port = 3000;

app.use(express.json());
app.use(express.static('public'));


// In-memory "database"
let items = [
  { id: 1, name: 'Item 1' },
  { id: 2, name: 'Item 2' }
];

// GET all items
app.get('/items', (req, res) => {
  res.json(items);
});

// GET single item
app.get('/items/:id', (req, res) => {
  const item = items.find(i => i.id === parseInt(req.params.id));
  if (!item) return res.status(404).send('Item not found');
  res.json(item);
});

// POST new item
app.post('/items', (req, res) => {
  const newItem = {
    id: items.length + 1,
    name: req.body.name
  };
  items.push(newItem);
  res.status(201).json(newItem);
});

// PUT update item
app.put('/items/:id', (req, res) => {
  const item = items.find(i => i.id === parseInt(req.params.id));
  if (!item) return res.status(404).send('Item not found');
  if (!req.body.name || typeof req.body.name !== 'string') {
    return res.status(400).send('Invalid name');
  }
  item.name = req.body.name;
  res.json(item);
});

// DELETE item
app.delete('/items/:id', (req, res) => {
  const index = items.findIndex(i => i.id === parseInt(req.params.id));
  if (index === -1) return res.status(404).send('Item not found');
  const deleted = items.splice(index, 1);
  res.json(deleted[0]);
});

app.listen(port, () => {
  console.log(`API running at http://localhost:${port}`);
});

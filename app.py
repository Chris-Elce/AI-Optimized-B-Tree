import streamlit as st
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

# ---- B-Tree Node and Cache Class Definitions ----

class BTreeNode:
    def __init__(self, leaf=False):
        self.leaf = leaf
        self.keys = []
        self.children = []

class BTreeWithCache:
    def __init__(self, t):
        self.root = BTreeNode(True)  # Initially an empty leaf node
        self.t = t
        self.cache = {}  # AI-powered node cache

    def insert(self, key):
        """Insert a key into the B-tree and update the cache if necessary"""
        root = self.root
        if len(root.keys) == (2 * self.t) - 1:
            new_root = BTreeNode(False)
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root

        self._insert_non_full(self.root, key)

        # AI decides if this key should be cached
        should_cache = cache_model.predict([[key, key_counts.get(key, 0)]])[0]
        if should_cache == 1:
            self.cache[key] = self.search(key)  # Store node in cache

    def _insert_non_full(self, node, key):
        """Insert a key into a non-full node"""
        i = len(node.keys) - 1
        if node.leaf:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            node.keys.insert(i + 1, key)
        else:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            if len(node.children[i].keys) == (2 * self.t) - 1:
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            self._insert_non_full(node.children[i], key)

    def _split_child(self, parent, index):
        """Split a full child node"""
        full_child = parent.children[index]
        new_node = BTreeNode(full_child.leaf)
        mid_index = self.t - 1
        parent.keys.insert(index, full_child.keys[mid_index])
        parent.children.insert(index + 1, new_node)

        new_node.keys = full_child.keys[mid_index + 1:]
        full_child.keys = full_child.keys[:mid_index]

        if not full_child.leaf:
            new_node.children = full_child.children[self.t:]
            full_child.children = full_child.children[:self.t]

    def search(self, key, node=None):
        """Search for a key, using cache if available"""
        if key in self.cache:
            return f"Cache Hit for key: {key}"

        if node is None:
            node = self.root

        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1

        if i < len(node.keys) and key == node.keys[i]:
            return f"Key {key} Found"

        if node.leaf:
            return f"Key {key} Not Found"

        return self.search(key, node.children[i])

    def display(self, node=None, level=0):
        """Display the B-tree structure in Streamlit"""
        if node is None:
            node = self.root

        st.markdown(f"**Level {level}:** `{node.keys}`")  # Show the keys at the current level
        for child in node.children:
            self.display(child, level + 1)

# ---- AI Model for Caching ----

def generate_workload(num_queries=2000, key_range=100):
    workload = [random.randint(1, key_range) for _ in range(num_queries)]
    df = pd.DataFrame(workload, columns=["key_accessed"])
    return df

# Create workload dataset
workload_df = generate_workload()
key_counts = workload_df["key_accessed"].value_counts().to_dict()
workload_df["access_count"] = workload_df["key_accessed"].map(key_counts)

# Label data for caching (1 = cache, 0 = no cache)
threshold = 30
workload_df["cache"] = workload_df["access_count"].apply(lambda x: 1 if x > threshold else 0)

# Train AI model
X = workload_df[["key_accessed", "access_count"]]
y = workload_df["cache"]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
cache_model = RandomForestClassifier(n_estimators=50, random_state=42)
cache_model.fit(X_scaled, y)

# ---- Streamlit App UI ----
st.title("🌲 AI-Optimized B-Tree with Caching")
st.markdown("""
### Welcome to the AI-Powered B-Tree App!
This app demonstrates how **B-Trees** work and how **Artificial Intelligence** can optimize them with caching.
- **B-Trees**: A self-balancing tree data structure used in databases and file systems.
- **AI Caching**: Uses machine learning to predict frequently accessed keys for faster retrieval.

Try inserting keys and searching for them to see caching in action! 🚀
""")

# B-Tree Instance (Persistent across interactions)
if "btree_ai_cache" not in st.session_state:
    st.session_state.btree_ai_cache = BTreeWithCache(t=2)

btree_ai_cache = st.session_state.btree_ai_cache  # Use session state

key_input = st.number_input("🔑 Insert Key into B-Tree", min_value=1, step=1, help="Enter a number to insert into the B-tree.")
if st.button("Insert Key"):
    btree_ai_cache.insert(key_input)
    st.success(f"Inserted Key: {key_input}")

search_key = st.number_input("🔍 Search Key in B-Tree", min_value=1, step=1, help="Enter a number to search in the B-tree.")
if st.button("Search Key"):
    result = btree_ai_cache.search(search_key)
    if "Found" in result:
        st.success(result)
    elif "Cache Hit" in result:
        st.info(result)
    else:
        st.error(result)

st.subheader("📂 B-Tree Structure")
btree_ai_cache.display()

st.subheader("🗃 Cached Nodes")
st.write(f"**Cached Keys:** {list(btree_ai_cache.cache.keys())}")

# ---- Query Frequency Distribution ----
st.subheader("📊 Query Frequency Distribution")
st.markdown("This graph shows how frequently different keys are accessed. The AI model uses this data to optimize caching decisions.")

fig, ax = plt.subplots(figsize=(12, 6))
query_counts = workload_df["key_accessed"].value_counts().sort_index()
query_counts.plot(kind="bar", ax=ax, color="skyblue", edgecolor="black")

# Adjust x-axis tick marks to prevent overlap
step = 3  # Adjust the step to control tick spacing
xticks = query_counts.index[::step]  # Select every 'step' value

ax.set_xticks(xticks)  # Apply the selected ticks
ax.set_xticklabels(xticks, rotation=45, ha="right", fontsize=10)  # Rotate and adjust label size

plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.title("Key Access Frequency", fontsize=14, fontweight="bold")
st.pyplot(fig)

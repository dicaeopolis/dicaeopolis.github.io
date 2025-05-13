# 嗯造轮子

某数据结构课程不允许用STL，但我又挺喜欢用STL的，咋办？

🧐……🤓💡

反正我也用不上太多操作，不妨直接封装常用的：

（别想了，没有手撕红黑树）

## `std::vector`
```cpp
#include <cstdlib>
template <typename T>
class vector
{
    private:
        T* data;
        size_t _size;
        size_t _capacity;

        constexpr static double phi = 1.618;
        constexpr static int initial_data_cnt = 16;
        void memory_expand()
        {
            _capacity = static_cast<size_t>(phi * _capacity);
            T* tmp = static_cast<T*>(realloc(data, _capacity * sizeof(T)));
            if(tmp != NULL) data = tmp;
            else std::cerr << "Un able to reacllocate memory.\n";
        }
    public:
        vector() : _size(0), _capacity(0) { data = nullptr; }
        vector(int size) : _size(size), _capacity(size) { data = static_cast<T*>(malloc(size * sizeof(T))); }
        bool empty() { return !_size; }
        void push_back(T new_item)
        {
            if(_size == 0)
            {
                data = static_cast<T*>(malloc(initial_data_cnt * sizeof(T)));
                _capacity = initial_data_cnt;
            }
            else if(_size == (_capacity - 1)) // prevent end() pointing to unallocated memory.
                memory_expand();
            data[_size] = new_item;
            ++_size;
        }
        const size_t size() { return _size; }
        T& operator[] (size_t idx) { return data[idx]; }
        T front() { return data[0]; }
        T back() { return data[_size - 1]; }
        T* begin() { return data; }
        T* end() { return data + _size; }
};
```

## 利用双向链表实现的`std::deque`、`std::queue`、`std::stack`和`std::list`

双向链表结点基础模板：
```cpp
template<typename T>
class _list
{
    public:
        _list<T>* _next;
        _list<T>* _prev;
        T _data;
        _list() : _next(nullptr), _prev(nullptr) { }
        _list(T data) : _data(data), _next(nullptr), _prev(nullptr) { }
        void insert_after(T data)
        {
            
            _list<T>* node = new _list<T>(data);
            node->_next = _next;
            node->_prev = this;
            _next->_prev = node;
            _next = node;
            
        }
        void insert_before(T data)
        {
            
            _list<T>* node = new _list<T>(data);
            node->_next = this;
            node->_prev = _prev;
            _prev->_next = node;
            _prev = node;
            
        }
};

template<typename T>
class _deque
{
    public:
        _list<T>* _head;
        _list<T>* _tail;
        _deque() : _head(), _tail()
        {
            _head = new _list<T>;
            _tail = new _list<T>;
            _head->_next = _tail;
            _tail->_prev = _head;
        }
        bool empty() { return (_head->_next == _tail && _tail->_prev == _head); }
        void push_front(T data) { _head->insert_after(data); }
        void push_back(T data) { _tail->insert_before(data); }
        void pop_front()
        {
            if(empty()) return ;
            _list<T>* next = _head->_next;
            next->_next->_prev = _head;
            _head->_next = next->_next;
            delete next;
        }
        void pop_back()
        {
            if(empty()) return ;
            _list<T>* prev = _tail->_prev;
            prev->_prev->_next = _tail;
            _tail->_prev = prev->_prev;
            delete prev;
        }
};
```

注意这里我尽量使得双向链表操作局限在三个节点的窗口里面。删除节点的操作需要有一个头节点来顶住，不然节点一删，就找不着北了。

### `deque`实现

```cpp
template<typename T>
class deque
{
    private:
        _deque<T> _data;
        size_t _size;
    public:
        deque() : _data() {  _size = 0; }
        bool empty() { return !_size && _data.empty(); }
        T front()
        {
            if(_data._head->_next != nullptr && _size != 0)
                return _data._head->_next->_data;
            std::cerr << "No front\n";
            T res;
            return res;
        }
        T back()
        {
            if(_data._tail->_prev != nullptr && _size != 0)
                return _data._tail->_prev->_data;
            std::cerr << "No back\n";
            T res;
            return res;
        }
        void push_front(T data) { _data.push_front(data); ++_size; }
        void push_back(T data) { _data.push_back(data); ++_size; }
        void pop_front()
        {
            if(_size != 0)
            {
                _data.pop_front();
                --_size;
            }
        }
        void pop_back()
        {
            if(_size != 0)
            {
                _data.pop_back();
                --_size;
            }
        }
};
```

### `queue`实现

```cpp
template<typename T>
class queue
{
    private:
        deque<T> _data;
    public:
        queue() : _data() {}
        bool empty() { return _data.empty(); }
        void push(T data) { _data.push_back(data); }
        void pop() { _data.pop_front(); }
        T front() { return _data.front(); }
};
```

### `stack`实现

```cpp
template<typename T>
class stack
{
    private:
        deque<T> _data;
    public:
        stack() : _data() {}
        bool empty() { return _data.empty(); }
        void push(T data) { _data.push_back(data); }
        void pop() { _data.pop_back(); }
        T top() { return _data.back(); }
};
```
### `list`实现

```cpp
template<typename T>
class list
{
    private:
        _list<T>* _head;
        _list<T>* _tail;
    public:
        class iterator
        {
            private:
                _list<T>* ptr;
            public:
                iterator(_list<T> p = nullptr) : ptr(p) {}
                iterator& operator++() { ptr = ptr->_next; return *this; }
                iterator& operator--() { ptr = ptr->_prev; return *this; }
                bool operator!=(const iterator& other) { return other.ptr != ptr; }
                T& operator*() { return ptr->_data; }
        };
        list() : _head(), _tail()
        {
            _head = new _list<T>;
            _tail = new _list<T>;
            _head->_next = _tail;
            _tail->_prev = _head;
        }
        bool empty() { return (_head->_next == _tail && _tail->_prev == _head); }
        void push_front(T data) { _head->insert_after(data); }
        void push_back(T data) { _tail->insert_before(data); }
        void pop_front()
        {
            if(empty()) return ;
            _list<T>* next = _head->_next;
            next->_next->_prev = _head;
            _head->_next = next->_next;
            delete next;
        }
        void pop_back()
        {
            if(empty()) return ;
            _list<T>* prev = _tail->_prev;
            prev->_prev->_next = _tail;
            _tail->_prev = prev->_prev;
            delete prev;
        }
        iterator begin() { return iterator(_head->_next); }
        iterator end() { return iterator(_tail); }
        void erase(iterator item)
        {
            _list<T>* node = item.ptr;
            if(node == _head || node == _tail) return;
            node->_prev->_next = node->_next;
            node->_next->_prev = node->_prev;
            delete node;
        }
};
```

这里自定义了迭代器类，供顺序访问使用。

## 哈希表

利用CRC64进行哈希并在编译期计算了系数表。

依赖前面的`list`和`vector`，当然也可以使用现成的STL。

```cpp
#include <array>
namespace crc64 {    
    constexpr uint64_t CRC64_POLY = 0x42F0E1EBA9EA3693ULL;

    constexpr std::array<uint64_t, 256> generate_crc64_table()
    {
        std::array<uint64_t, 256> table = {};
        for (int i = 0; i < 256; ++i) {
            uint64_t crc = i;
            for (int j = 0; j < 8; ++j)
                crc = (crc & 1) ? (crc >> 1) ^ CRC64_POLY : crc >> 1;
            table[i] = crc;
        }
        return table;
    }

    constexpr std::array<uint64_t, 256> crc64_table = generate_crc64_table();

    uint64_t crc64(const uint8_t *data, size_t length)
    {
        uint64_t crc = 0xFFFFFFFFFFFFFFFFULL;
        for (size_t i = 0; i < length; i++) {
            uint8_t index = (uint8_t)(crc ^ data[i]);
            crc = (crc >> 8) ^ crc64_table[index];
        }
        return crc ^ 0xFFFFFFFFFFFFFFFFULL;
    }
}


constexpr size_t hash_mod = 126271;

template<typename key_type, typename value_type>
class unordered_map {
    typedef std::pair<key_type, value_type> mapped_type;
    private:
        vector<list<mapped_type>> table;
        size_t pair_cnt = 0;
        size_t get_idx(const key_type& key)
        {
            auto hash = crc64::crc64((uint8_t*) &key, sizeof(key));
            return static_cast<size_t>(hash % hash_mod);
        }
        const size_t get_idx(const key_type& key) const
        {
            auto hash = crc64::crc64((uint8_t*) &key, sizeof(key));
            return static_cast<size_t>(hash % hash_mod);
        }
        auto& get_item(const key_type& key)
        {
            auto idx = get_idx(key);
            return table[idx];
        }
        const auto& get_item(const key_type& key) const
        {
            auto idx = get_idx(key);
            return table[idx];
        }
    public:
        unordered_map() : table(hash_mod) {}
        value_type& operator[](const key_type& key)
        {
            auto& item = get_item(key);
            for(auto& val : item)
                if(val.first == key)
                    return val.second;
            value_type val;
            item.push_back(std::make_pair(key, val));
            ++pair_cnt;
            return item.back().second;
        }
        bool empty() const noexcept
        {
            return !(pair_cnt);
        }
        bool find(const key_type &key) const noexcept
        {
            const auto& item = get_item(key);
            for(const auto& val : item)
                if(val.first == key)
                    return true;
            return false;
        }
        void erase(const key_type& key)
        {
            auto& item = get_item(key);
            for(auto it = item.begin(); it != item.end(); ++it)
                if(it->first == key) {
                    item.erase(it);
                    --pair_cnt;
                    break;
                }
        }      
};
```

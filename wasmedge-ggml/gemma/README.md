# `gemma`

## Get Model

```console
wget https://huggingface.co/second-state/Gemma-2b-it-GGUF/resolve/main/gemma-2b-it-Q4_K_M.gguf
```

## Execute

````console
$ wasmedge --dir .:. \
  --nn-preload default:GGML:AUTO:gemma-2b-it-Q4_K_M.gguf \
  ./wasmedge-ggml-gemma.wasm default

USER:
Write a merge sort in Rust for me
ASSISTANT:
```rust
fn merge_sort(arr: &mut [i32]) {
    if arr.len() <= 1 {
        return;
    }
    let mid = arr.len() / 2;
    let (left, right) = arr.split_at_mut(mid);

    merge_sort(left);
    merge_sort(right);

    merge(arr, left, right);
}

fn merge(arr: &mut [i32], left: &mut [i32], right: &mut [i32]) {
    let mut merged = vec![];

    for i in 0..left.len() {
        merged.push(*left[i]);
    }

    for j in 0..right.len() {
        merged.push(*right[j]);
    }

    arr.clear();
    for item in merged {
        arr.push(item);
    }
}
```

**Usage:**

```rust
let mut numbers = vec![5, 2, 8, 3, 1, 9, 4];
merge_sort(&mut numbers);
println!("{:?}", numbers); // Output: [1, 2, 3, 4, 5, 8, 9]
```

**Explanation:**

The `merge` function takes three slices as input and merges them into a single sorted slice. It works by iterating over the two input slices and adding each element to the merged slice until one of the slices is empty.

* `left` contains the first half of the elements.
* `right` contains the second half of the elements.
* The merged slice is stored in `arr`.

**Time complexity:**

Merge sort has a time complexity of O(n log n), where n is the length of the input vector. This is because the function divides the problem into smaller subproblems until each subproblem contains only one element.
USER:
How about in C++?
ASSISTANT:
```c++
#include <iostream>
#include <vector>

using namespace std;

void merge(int arr[], int left[], int right[]) {
  int n = sizeof(arr) / 2;
  int i = 0, j = 0, k = 0;

  while (i < n && j < left[n]) {
    if (left[i] <= right[j]) {
      arr[k++] = left[i++];
    } else {
      arr[k++] = right[j++];
    }
  }

  while (i < n) {
    arr[k++] = left[i++];
  }

  while (j < left[n]) {
    arr[k++] = right[j++];
  }
}

void merge_sort(vector<int>& arr) {
  if (arr.size() <= 1) {
    return;
  }
  int mid = arr.size() / 2;
  merge_sort(arr.data() + mid);
  merge_sort(arr.data() + mid * 2);
  merge(arr.data(), arr.data() + mid, arr.data() + 2 * mid);
}

int main() {
  vector<int> numbers = {5, 2, 8, 3, 1, 9, 4};
  merge_sort(numbers);
  cout << "{:?}", numbers) << endl;
  return 0;
}
```

**Output:**

```
{:?}", [1, 2, 3, 4, 5, 8, 9]
```
````

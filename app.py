import streamlit as st

def add_text_inputs():
    col1, col2 = st.columns(2)
    with col1:
        text_input1 = st.text_input('Enter standard text here:')
    with col2:
        text_input2 = st.text_input('Enter non-standard text here:')
    return text_input1, text_input2

def main():
    st.title('Dynamic Text Input Boxes (Side by Side)')

    num_pairs = st.sidebar.number_input('Number of Text Input Pairs', min_value=1, max_value=10, value=1)

    text_inputs = []
    for _ in range(num_pairs):
        text_inputs.append(add_text_inputs())

    if st.button('Add Text Input Pair'):
        text_inputs.append(add_text_inputs())

    st.write('## Text Input Pairs:')
    for i, pair in enumerate(text_inputs):
        st.write(f'Text Input Pair {i+1}: {pair[0]} - {pair[1]}')

if __name__ == "__main__":
    main()

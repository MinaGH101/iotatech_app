import streamlit as st
import streamlit.components.v1 as components
# st.title('🎈 Presentation Deck')
# components.iframe("https://docs.google.com/presentation/d/1gUZdJFerqnd-189PXmZh33j3eavwaF2a/edit?usp=sharing&ouid=111224514069246419633&rtpof=true&sd=true", height=480)
from barfi.flow import Block, SchemaManager, ComputeEngine
from barfi.flow.streamlit import st_flow


number_block = Block(name="Number")
number_block.add_output(name="Output 1")
number_block.add_option(
    name="display-option", type="display", value="This is a Block with Number option."
)
number_block.add_option(name="number-block-option", type="number")
 
def number_block_func(self):
    number_value = self.get_option(name="number-block-option")
    self.set_interface(name="Output 1", value=number_value)
 
number_block.add_compute(number_block_func)

result_block = Block(name="Result")
result_block.add_input(name="Input 1")
result_block.add_option(
    name="display-option", type="display", value="This is a Result Block."
)
 
def result_block_func(self):
    number_value = self.get_interface(name="Input 1")
    print(number_value)
 
result_block.add_compute(result_block_func)

base_blocks=[number_block, result_block]
barfi_result = st_flow(base_blocks)
 
# 05: You can view the schema here
st.write(barfi_result)


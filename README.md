**Creating a Compile Debug Tool for PyTorch 2.0**

Zhengfei Gong, Haiyu Wei, Patrick Yeh

This project focuses on building a debugging tool to enhance the development experience when working with PyTorch 2.0 and its torch.compile feature. 
The tool is designed to identify and explain graph breaks in dynamic computation graphs, a common issue in machine learning model optimization. 
By leveraging PyTorch profiling tools, custom scripts, and large language models (LLMs), the tool provides actionable insights, detailed explanations, 
and targeted suggestions to resolve graph compilation errors efficiently. Our work combines structured error classification with LLM-powered feedback, 
enabling developers to address issues with greater clarity and reduced effort.

The code repository contains graphbreak-taxonomy.ipynb, which contains our work in detailing a taxonomy of graph breaks,
Graph_Detector_LLM_Explanation_API_Configured.ipynb, which employs the Gemini API to prompt engineer and process the
explanations provided by PyTorch, and graphbreak-taxonomy.pdf, which provides a detailed list of our taxonomy results.
Graph_Detector_LLM_Explanation_FINAL_PRESENTATION_DEMO.ipynb is a notebook that has a demo which showcases some of the
key features underlying our analysis of PyTorch graph breaks.

To execute the code, one can run the notebooks on Google Colab. We used the T4 GPU when testing. Also, for the Gemini
API, one would need to insert a Gemini API key in the relevant section of the code (which is clearly commented), but to
generate the key, one needs to be using their personal Gmail account instead of their Columbia account.

An example command would be to run the LLM explanation notebook, fill out "EXAMPLE_KEY" in the field for the Gemini set_key()
call, choose the T4 GPU runtime, and then run all.

Results and observations of our taxonomy are listed in the file graphbreak-taxonomy.pdf. One may also view the document
in the following Google Doc: https://docs.google.com/document/d/13BO5mU-wegXnhewIak9msZUc1IbFlDntuUCgNmz5fUQ/edit?usp=sharing

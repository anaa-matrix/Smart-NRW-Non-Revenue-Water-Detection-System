with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace old function with updated version
old_func = '''def generate_scalogram(self, ae_signal, scales=64):
        \"\"\"Generate scalogram for Stage 2 CNN analysis\"\"\"
        
        # CWT with Morlet wavelet (research-validated choice)
        scale_range = np.logspace(np.log10(1), np.log10(scales), scales)'''

new_func = '''def generate_scalogram(self, ae_signal, scales=32):
        \"\"\"Generate scalogram for Stage 2 CNN analysis\"\"\"
        
        # Truncate signal to prevent memory error
        max_samples = 5000
        if len(ae_signal) > max_samples:
            ae_signal = ae_signal[:max_samples]
        
        # CWT with Morlet wavelet (research-validated choice)  
        scale_range = np.logspace(np.log10(1), np.log10(scales), scales)'''

content = content.replace(old_func, new_func)

# Replace CNN training sample size
content = content.replace('min(1000, len(st.session_state.ae_data))', 'min(50, len(st.session_state.ae_data))')

# Write back using UTF-8 encoding
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed memory issues!')

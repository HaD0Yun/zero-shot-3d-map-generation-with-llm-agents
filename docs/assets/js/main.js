document.addEventListener('DOMContentLoaded', () => {
    
    // --- Tab Switching Logic ---
    const tabs = document.querySelectorAll('.tabs li');
    const tabContentBoxes = document.querySelectorAll('.tab-content-box');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class from all tabs
            tabs.forEach(item => item.classList.remove('is-active'));
            
            // Add active class to clicked tab
            tab.classList.add('is-active');

            // Hide all tab content
            tabContentBoxes.forEach(box => box.style.display = 'none');

            // Show target content
            const target = tab.dataset.tab;
            document.getElementById(target).style.display = 'block';
        });
    });

    // --- Copy to Clipboard Logic ---
    const copyButtons = document.querySelectorAll('.copy-button');

    copyButtons.forEach(button => {
        button.addEventListener('click', async () => {
            const pre = button.parentElement;
            const code = pre.querySelector('code');
            const text = code.innerText;

            try {
                await navigator.clipboard.writeText(text);
                
                // Visual feedback
                const originalIcon = button.innerHTML;
                button.innerHTML = '<span class="icon"><i class="fas fa-check"></i></span>';
                
                setTimeout(() => {
                    button.innerHTML = originalIcon;
                }, 2000);
            } catch (err) {
                console.error('Failed to copy text: ', err);
            }
        });
    });

    // --- Smooth Scroll for Anchor Links ---
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // --- Initialize first tab ---
    if(tabs.length > 0) {
        tabs[0].click();
    }
});

// --- Copy Full Prompt Function (Global) ---
async function copyFullPrompt() {
    const promptElement = document.getElementById('full-prompt');
    const code = promptElement.querySelector('code');
    const text = code.innerText;
    
    try {
        await navigator.clipboard.writeText(text);
        
        // Find the button
        const btn = document.querySelector('.copy-prompt-btn');
        const originalHTML = btn.innerHTML;
        
        btn.innerHTML = '<span class="icon"><i class="fas fa-check"></i></span><span>Copied!</span>';
        btn.classList.remove('is-primary');
        btn.classList.add('is-success');
        
        setTimeout(() => {
            btn.innerHTML = originalHTML;
            btn.classList.remove('is-success');
            btn.classList.add('is-primary');
        }, 2000);
    } catch (err) {
        console.error('Failed to copy prompt: ', err);
        alert('Failed to copy. Please select and copy manually.');
    }
}

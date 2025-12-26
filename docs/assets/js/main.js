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

/*
 * Linker script for the pure-Rust ET-SoC-1 compute kernel.
 *
 * The kernel is position-dependent and must be linked at the fixed U-mode entry
 * address, which coincides with the base of the host-managed DRAM region. This
 * mirrors the SDK test drive's sections.ld. No .bss is permitted in a kernel
 * (firmware does not zero it), which the assertion at the end enforces.
 */

KERNEL_UMODE_ENTRY = 0x8005801000;

ENTRY(_start)

SECTIONS
{
    . = KERNEL_UMODE_ENTRY;

    .text :
    {
        *(.text.init)   /* _start first */
        *(.text .text.*)
    }

    .rodata :
    {
        *(.rodata .rodata.*)
        *(.srodata .srodata.*)
    }

    .data :
    {
        *(.data .data.*)
    }

    .sdata :
    {
        __global_pointer$ = . + 0x800;
        *(.sdata .sdata.*)
    }

    .bss (NOLOAD) :
    {
        __bss_start = .;
        *(.bss .bss.* .sbss .sbss.*)
        __bss_end = .;
    }

    /DISCARD/ :
    {
        *(.riscv.attributes)
        *(.comment)
        *(.eh_frame)
        *(.note .note.*)
    }
}

ASSERT((__bss_start == __bss_end), "kernel must not contain .bss")

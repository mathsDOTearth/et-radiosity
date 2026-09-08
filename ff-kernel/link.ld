/*
 * Linker script for the ff-kernel ET-SoC-1 compute kernel.
 *
 * The kernel binary is position-dependent and must be linked at the fixed
 * U-mode entry address that the host runtime maps for kernel code. This
 * address mirrors the SDK's sections.ld convention and is required to match
 * the address used by Device::load_kernel on the host side.
 *
 * No .bss section is permitted: the firmware does not zero-initialise device
 * DRAM before kernel entry. The assertion at the end enforces this.
 */

KERNEL_UMODE_ENTRY = 0x8005801000;

ENTRY(_start)

SECTIONS
{
    . = KERNEL_UMODE_ENTRY;

    /* _start (from kernel_entry!) must appear first so the ELF entry address
     * resolves correctly. */
    .text :
    {
        *(.text.init)
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

    /* __global_pointer$ must lie within +/- 2 KiB of all .sdata/.sbss symbols
     * for linker relaxation to generate gp-relative loads/stores. */
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

ASSERT((__bss_start == __bss_end), "kernel must not contain .bss: initialise all data in .data or compute it at runtime")
